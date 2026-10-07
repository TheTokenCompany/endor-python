"""Run: one adapter on one base model inside a project, driven step by step.

    run = project.runs.create(base_model="decider-2b", rank=16)
    fb = run.forward_backward(datums)                 # gradients accumulate
    opt = run.optim_step(learning_rate=1e-4)          # AdamW step, then zero gradients
    fb.result(); opt.result()                         # submit both, then wait: no idle round trip
    model = run.save_checkpoint("v1").result()        # "tickets/v1", usable with client.system_one at once

Every compute or save call carries a sequence number. The trainer executes calls strictly in that order, and the
server accepts a call only when its number is the next one it expects. A ``Run`` handle starts from the server's
``next_seq_id``, holds its lock from assigning a number until the request returns, and advances only on success,
so a retried request reuses its number and body. Two handles driving the same run see ``ConflictError`` with code
``seq_conflict``: that is intended.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from . import _constants as C
from ._http import Transport
from ._log import logger
from .futures import APIFuture
from .types import (
    AdamParams,
    Datum,
    Evaluation,
    ForwardOutput,
    LossFn,
    MetricPoint,
    OptimStepOutput,
    RunInfo,
    option_keys,
)

__all__ = ["Run", "MAX_DATUMS_PER_CALL"]

_FINAL = ("closed", "failed")
_MODEL_NAME = re.compile(C.SDK_MODEL_NAME)
_CLOSE_POLL_S = 1.0

MAX_DATUMS_PER_CALL = C.MAX_DATUMS_PER_CALL
"""Bigger batches are split into several requests; the merged result and the gradients are the same."""


class Run:
    """A training run. Get one from ``project.runs.create(...)``, ``project.runs.get(id)`` or
    ``project.runs.list()``.

    A run holds a GPU, billed per GPU-hour, until it is closed or idle for 15 minutes, and an org can have at most
    4 runs that aren't closed (idle ones included). Use it as a context manager so it is closed even on errors::

        with project.runs.create(base_model="jev-9b") as run:
            ...
    """

    def __init__(self, transport: Transport, info: dict[str, Any]) -> None:
        self._t = transport
        self.info_ = RunInfo.model_validate(info)
        """The run as last fetched from the server (``run.info()`` refreshes it)."""
        self.id: str = self.info_.id
        self.project: str = self.info_.project
        self.ready: APIFuture[Any] = (
            APIFuture(transport, self.info_.ready_future_id, method="run.ready")
            if self.info_.ready_future_id
            else APIFuture.completed({})
        )
        """Resolves when the run's trainer has the base model loaded and the adapter ready."""
        self._seq = self.info_.next_seq_id
        self._lock = threading.Lock()
        self._alock: asyncio.Lock | None = None
        self._last_optim_seq: int | None = None
        self._last_save_seq: int | None = None

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, project={self.project!r}, base_model={self.info_.base_model!r})"

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._close_on_exit(exc is not None)

    async def __aenter__(self) -> Run:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc is None:
            await self.close_async()
            return
        try:
            await self.close_async()
        except Exception:
            logger.warning("run %s: close failed; close it with `endor runs close %s`", self.id, self.id, exc_info=True)

    def _close_on_exit(self, failing: bool) -> None:
        """Close; while another exception is propagating, a failed close is logged instead of hiding it."""
        if not failing:
            self.close()
            return
        try:
            self.close()
        except Exception:
            logger.warning("run %s: close failed; close it with `endor runs close %s`", self.id, self.id, exc_info=True)

    @property
    def next_seq_id(self) -> int:
        """The sequence number this handle will give its next compute or save call."""
        return self._seq

    # ------------------------------------------------------------------ compute
    def forward(self, data: Iterable[Datum], loss_fn: LossFn = "cross_entropy") -> APIFuture[ForwardOutput]:
        """Every option's probability and each datum's loss with the run's current adapter. No gradients.

        Use it to score held-out data during training without saving a model. Billed like training.
        """
        return self._fb("forward", data, loss_fn)

    def forward_backward(self, data: Iterable[Datum], loss_fn: LossFn = "cross_entropy") -> APIFuture[ForwardOutput]:
        """Loss and gradients, accumulated until the next ``optim_step``.

        The update ``optim_step`` applies is the weight-averaged gradient over every datum since the previous step,
        so splitting a batch across calls changes nothing. Batches over ``MAX_DATUMS_PER_CALL`` are split for you.
        """
        return self._fb("forward_backward", data, loss_fn)

    async def forward_async(self, data: Iterable[Datum], loss_fn: LossFn = "cross_entropy") -> APIFuture[ForwardOutput]:
        """``forward`` for async code: submits without blocking; ``await`` the returned future for the result."""
        return await self._fb_async("forward", data, loss_fn)

    async def forward_backward_async(
        self, data: Iterable[Datum], loss_fn: LossFn = "cross_entropy"
    ) -> APIFuture[ForwardOutput]:
        """Async submit; ``await`` the returned future for the result."""
        return await self._fb_async("forward_backward", data, loss_fn)

    def optim_step(
        self, adam_params: AdamParams | None = None, *, learning_rate: float | None = None
    ) -> APIFuture[OptimStepOutput]:
        """Apply the accumulated gradients with AdamW and zero them. Pass the learning rate every step: the schedule
        is yours. Fails with ``no_gradients`` when nothing was accumulated."""
        body = self._optim_body(adam_params, learning_rate)
        r = self._submit("optim_step", body, kind="optim")
        return APIFuture(self._t, r["future_id"], OptimStepOutput.model_validate, method="run.optim_step")

    async def optim_step_async(
        self, adam_params: AdamParams | None = None, *, learning_rate: float | None = None
    ) -> APIFuture[OptimStepOutput]:
        """``optim_step`` for async code."""
        body = self._optim_body(adam_params, learning_rate)
        r = await self._submit_async("optim_step", body, kind="optim")
        return APIFuture(self._t, r["future_id"], OptimStepOutput.model_validate, method="run.optim_step")

    # ------------------------------------------------------------------ models and lifecycle
    def save_checkpoint(
        self,
        name: str,
        *,
        include_optimizer: bool = False,
        ttl_seconds: int | None = None,
        user_metadata: dict[str, Any] | None = None,
    ) -> APIFuture[str]:
        """Save the adapter as the project model ``"<project>/<name>"``. The future resolves to that id, which
        ``client.system_one(model=...)`` accepts at once.

        ``name``: lowercase letters, digits, ``.``, ``_`` and ``-``, up to 63 characters, starting with a letter
        (names starting with a digit are continuous learning's), and not ``base`` (the project's base model).
        Raises ``ValueError`` otherwise, before anything is sent.

        ``include_optimizer`` also stores the optimizer state, so
        ``runs.create(from_model=..., include_optimizer=True)``
        resumes exactly. ``ttl_seconds`` (1 hour to 10 years) deletes the model after that time; None keeps it.
        """
        body = self._save_body(name, include_optimizer, ttl_seconds, user_metadata)
        r = self._submit("save_checkpoint", body, kind="save")
        return APIFuture(self._t, r["future_id"], lambda x: str(x["model"]), method="run.save_checkpoint")

    async def save_checkpoint_async(
        self,
        name: str,
        *,
        include_optimizer: bool = False,
        ttl_seconds: int | None = None,
        user_metadata: dict[str, Any] | None = None,
    ) -> APIFuture[str]:
        """``save_checkpoint`` for async code."""
        body = self._save_body(name, include_optimizer, ttl_seconds, user_metadata)
        r = await self._submit_async("save_checkpoint", body, kind="save")
        return APIFuture(self._t, r["future_id"], lambda x: str(x["model"]), method="run.save_checkpoint")

    def close(self, *, wait: bool = False, timeout: float | None = None) -> RunInfo:
        """Close the run: later calls are refused, calls already accepted still finish, then the GPU is released
        and billing stops. Without ``wait`` it returns at once, with status ``closing`` while accepted calls finish
        (``closed`` when nothing is pending); with ``wait=True`` it polls until the run is ``closed`` (or
        ``failed``), for at most ``timeout`` seconds. Unsaved progress is lost. Idempotent; a ``with`` block calls
        it for you.

        Logs a warning when this handle stepped the optimizer after its last save.
        """
        self._warn_unsaved()
        self.info_ = RunInfo.model_validate(
            self._t.request("POST", f"/v1/runs/{self.id}/close", method_name="run.close")
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        while wait and self.info_.status not in _FINAL and (deadline is None or time.monotonic() < deadline):
            time.sleep(_CLOSE_POLL_S)
            self.info()
        return self.info_

    async def close_async(self, *, wait: bool = False, timeout: float | None = None) -> RunInfo:
        """``close`` for async code."""
        self._warn_unsaved()
        self.info_ = RunInfo.model_validate(
            await self._t.arequest("POST", f"/v1/runs/{self.id}/close", method_name="run.close")
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        while wait and self.info_.status not in _FINAL and (deadline is None or time.monotonic() < deadline):
            await asyncio.sleep(_CLOSE_POLL_S)
            self.info_ = RunInfo.model_validate(
                await self._t.arequest("GET", f"/v1/runs/{self.id}", method_name="run.info")
            )
        return self.info_

    def info(self) -> RunInfo:
        """Refresh and return the run's status, step and ``next_seq_id``."""
        self.info_ = RunInfo.model_validate(self._t.request("GET", f"/v1/runs/{self.id}", method_name="run.info"))
        return self.info_

    # ------------------------------------------------------------------ dashboard
    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record custom metrics (``{"heldout/accuracy": 0.91}``) at ``step`` (default: the run's current step on
        the server). They are plotted next to the automatic loss, learning-rate and gradient-norm curves."""
        body: dict[str, Any] = {"metrics": dict(metrics)}
        if step is not None:
            body["step"] = step
        self._t.request("POST", f"/v1/runs/{self.id}/metrics", json=body, method_name="run.log")

    def metrics(self, keys: Iterable[str] | None = None, since_step: int | None = None) -> list[MetricPoint]:
        """The run's metric series, automatic and custom, optionally filtered by key."""
        params = {"keys": ",".join(keys) if keys else None, "since_step": since_step}
        r = self._t.request("GET", f"/v1/runs/{self.id}/metrics", params=params, method_name="run.metrics")
        return [MetricPoint.model_validate(p) for p in r]

    def log_eval(
        self, model: str, results: dict[str, Any], *, step: int | None = None, name: str = "eval"
    ) -> Evaluation:
        """Record evaluation results you computed yourself (``endor.metrics.decision_metrics``) on this run."""
        body = {"model": model, "results": results, "step": step, "name": name}
        r = self._t.request(
            "POST", f"/v1/runs/{self.id}/evaluations", json=body, method_name="run.log_eval", idempotent=True
        )
        return Evaluation.model_validate(r)

    # ------------------------------------------------------------------ sequencing
    def _submit(self, op: str, body: dict[str, Any], *, kind: str) -> dict[str, Any]:
        """Assign the next sequence number, POST, and advance the counter only on success."""
        with self._lock:
            seq = self._seq
            r = self._t.request(
                "POST", f"/v1/runs/{self.id}/{op}", json={**body, "seq_id": seq}, method_name=f"run.{op}"
            )
            self._seq = seq + 1
            self._mark(kind, seq)
            return dict(r)

    async def _submit_async(self, op: str, body: dict[str, Any], *, kind: str) -> dict[str, Any]:
        async with self._async_lock():
            seq = self._seq
            r = await self._t.arequest(
                "POST", f"/v1/runs/{self.id}/{op}", json={**body, "seq_id": seq}, method_name=f"run.{op}"
            )
            self._seq = seq + 1
            self._mark(kind, seq)
            return dict(r)

    def _async_lock(self) -> asyncio.Lock:
        if self._alock is None:
            self._alock = asyncio.Lock()
        return self._alock

    def _mark(self, kind: str, seq: int) -> None:
        if kind == "optim":
            self._last_optim_seq = seq
        elif kind == "save":
            self._last_save_seq = seq

    def _warn_unsaved(self) -> None:
        if self._last_optim_seq is not None and (
            self._last_save_seq is None or self._last_save_seq < self._last_optim_seq
        ):
            logger.warning(
                "run %s: closing with optimizer steps newer than the last save_checkpoint; that progress is lost",
                self.id,
            )

    # ------------------------------------------------------------------ forward / forward_backward
    @staticmethod
    def _chunks(data: Iterable[Datum], loss_fn: LossFn) -> list[dict[str, Any]]:
        datums = list(data)
        if not datums:
            raise ValueError("no data: pass at least one Datum")
        for i, d in enumerate(datums):  # every chunk is checked before any is sent, so a batch is never half sent
            check_datum(d, f"data[{i}]")
        n = MAX_DATUMS_PER_CALL
        return [
            {"data": [d.to_wire() for d in datums[i : i + n]], "loss_fn": loss_fn} for i in range(0, len(datums), n)
        ]

    def _fb_future(self, op: str, refs: list[dict[str, Any]]) -> APIFuture[ForwardOutput]:
        parse: Callable[[Any], ForwardOutput] = ForwardOutput.model_validate
        futures = [APIFuture(self._t, r["future_id"], parse, method=f"run.{op}") for r in refs]
        if len(futures) == 1:
            return futures[0]
        return APIFuture(self._t, children=list(futures), combine=merge_outputs, method=f"run.{op}")

    def _fb(self, op: str, data: Iterable[Datum], loss_fn: LossFn) -> APIFuture[ForwardOutput]:
        chunks = self._chunks(data, loss_fn)
        with self._lock:  # the chunks of one call get consecutive sequence numbers
            refs: list[dict[str, Any]] = []
            try:
                for body in chunks:
                    seq = self._seq
                    r = self._t.request(
                        "POST", f"/v1/runs/{self.id}/{op}", json={**body, "seq_id": seq}, method_name=f"run.{op}"
                    )
                    self._seq = seq + 1
                    refs.append(dict(r))
            except BaseException:
                self._cancel_partial(op, refs, len(chunks))
                raise
        return self._fb_future(op, refs)

    async def _fb_async(self, op: str, data: Iterable[Datum], loss_fn: LossFn) -> APIFuture[ForwardOutput]:
        chunks = self._chunks(data, loss_fn)
        async with self._async_lock():
            refs: list[dict[str, Any]] = []
            try:
                for body in chunks:
                    seq = self._seq
                    r = await self._t.arequest(
                        "POST", f"/v1/runs/{self.id}/{op}", json={**body, "seq_id": seq}, method_name=f"run.{op}"
                    )
                    self._seq = seq + 1
                    refs.append(dict(r))
            except BaseException:
                self._cancel_partial(op, refs, len(chunks))
                raise
        return self._fb_future(op, refs)

    def _cancel_partial(self, op: str, refs: list[dict[str, Any]], total: int) -> None:
        """A later chunk was refused by the server (402, 429, ...): cancel the chunks it already accepted, so their
        gradients don't reach the next optim_step. A chunk that already started can't be cancelled; say so."""
        if not refs:
            return
        states = []
        for r in refs:
            try:
                f = self._t.request("POST", f"/v1/futures/{r['future_id']}/cancel", method_name=f"run.{op}")
                states.append(f.get("status"))
            except Exception:
                states.append("unknown")
        if any(s != "cancelled" for s in states):
            logger.warning(
                "run %s: %s was refused after %d of %d chunks were accepted; cancelling them left statuses %s, "
                "so some gradients may still be applied by the next optim_step",
                self.id,
                op,
                len(refs),
                total,
                states,
            )

    @staticmethod
    def _optim_body(adam_params: AdamParams | None, learning_rate: float | None) -> dict[str, Any]:
        p = adam_params or AdamParams()
        if learning_rate is not None:
            p = p.model_copy(update={"learning_rate": learning_rate})
        return {"adam_params": p.model_dump()}

    @staticmethod
    def _save_body(
        name: str, include_optimizer: bool, ttl_seconds: int | None, user_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        check_model_name(name)
        return {
            "name": name,
            "include_optimizer": include_optimizer,
            "ttl_seconds": ttl_seconds,
            "user_metadata": user_metadata or {},
        }


def check_model_name(name: str) -> None:
    """The API's rule for model names saved from the SDK; raises ValueError naming the problem."""
    if name == C.BASE_MODEL_NAME:
        raise ValueError('"base" is reserved for the project\'s base model; pick another model name')
    if not isinstance(name, str) or not _MODEL_NAME.fullmatch(name):
        raise ValueError(
            f"bad model name {name!r}: start with a letter (a-z), then lowercase letters, digits, '.', '_' or '-', "
            "63 characters at most (names starting with a digit are reserved for continuous learning)"
        )


def check_datum(d: Datum, where: str) -> None:
    """The server's datum rules, checked locally: the target is exactly one of ``label`` (an option id) or ``probs``
    (every option id, non-negative, summing to 1 ± 0.02). Raises ValueError naming ``where``."""
    keys = option_keys(d.question)
    t = d.target
    if (t.label is None) == (t.probs is None):
        raise ValueError(f"{where}: target needs exactly one of label or probs")
    if t.label is not None and t.label not in keys:
        raise ValueError(f"{where}: target.label {t.label!r} is not an option ({keys})")
    if t.probs is not None:
        if set(t.probs) != set(keys):
            raise ValueError(f"{where}: target.probs must cover exactly the options {keys}")
        if min(t.probs.values()) < 0 or not math.isclose(sum(t.probs.values()), 1.0, abs_tol=0.02):
            raise ValueError(f"{where}: target.probs must be non-negative and sum to 1")


def merge_outputs(outs: list[ForwardOutput]) -> ForwardOutput:
    """Combine the chunk results of one batch: outputs concatenated, metrics re-aggregated.

    ``loss:mean`` is weight-averaged over the whole batch, exactly as the trainer averages the gradients.
    """
    n = sum(o.metrics.get("n", len(o.outputs)) for o in outs)
    loss_sum = sum(o.metrics.get("loss:sum", 0.0) for o in outs)
    w_sum = sum(o.metrics.get("weight:sum", 0.0) for o in outs)
    correct = sum(o.metrics.get("accuracy", 0.0) * o.metrics.get("n", len(o.outputs)) for o in outs)
    return ForwardOutput(
        outputs=[x for o in outs for x in o.outputs],
        metrics={
            "loss:sum": loss_sum,
            "weight:sum": w_sum,
            "loss:mean": loss_sum / w_sum if w_sum else 0.0,
            "accuracy": correct / n if n else 0.0,
            "n": n,
        },
    )
