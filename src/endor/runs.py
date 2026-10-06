"""Run: one adapter on one base model inside a project, driven step by step.

    run = project.runs.create(base_model="pplx-decider-v1-27b", rank=16)
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
import threading
from collections.abc import Callable, Iterable
from typing import Any

from . import _constants as C
from ._http import Transport
from ._log import logger
from .futures import APIFuture
from .types import AdamParams, Datum, Evaluation, ForwardOutput, LossFn, MetricPoint, OptimStepOutput, RunInfo

__all__ = ["Run", "MAX_DATUMS_PER_CALL"]

MAX_DATUMS_PER_CALL = C.MAX_DATUMS_PER_CALL
"""Bigger batches are split into several requests; the merged result and the gradients are the same."""


class Run:
    """A training run. Get one from ``project.runs.create(...)``, ``project.runs.get(id)`` or
    ``project.runs.list()``."""

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

    def __exit__(self, *exc: object) -> None:
        self.close()

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
        body = self._save_body(name, include_optimizer, ttl_seconds, user_metadata)
        r = await self._submit_async("save_checkpoint", body, kind="save")
        return APIFuture(self._t, r["future_id"], lambda x: str(x["model"]), method="run.save_checkpoint")

    def close(self) -> RunInfo:
        """Release the run's trainer. Pending operations finish first; unsaved progress is lost. Idempotent.

        Logs a warning when this handle stepped the optimizer after its last save.
        """
        self._warn_unsaved()
        self.info_ = RunInfo.model_validate(
            self._t.request("POST", f"/v1/runs/{self.id}/close", method_name="run.close")
        )
        return self.info_

    async def close_async(self) -> RunInfo:
        self._warn_unsaved()
        self.info_ = RunInfo.model_validate(
            await self._t.arequest("POST", f"/v1/runs/{self.id}/close", method_name="run.close")
        )
        return self.info_

    def info(self) -> RunInfo:
        """Refresh and return the run's status, step and ``next_seq_id``."""
        self.info_ = RunInfo.model_validate(self._t.request("GET", f"/v1/runs/{self.id}", method_name="run.info"))
        return self.info_

    # ------------------------------------------------------------------ dashboard
    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record custom metrics (``{"heldout/accuracy": 0.91}``) at ``step`` (default: the run's current step).
        They are plotted next to the automatic loss, learning-rate and gradient-norm curves."""
        body = {"step": self.info_.step if step is None else step, "metrics": dict(metrics)}
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
            refs = []
            for body in chunks:
                seq = self._seq
                r = self._t.request(
                    "POST", f"/v1/runs/{self.id}/{op}", json={**body, "seq_id": seq}, method_name=f"run.{op}"
                )
                self._seq = seq + 1
                refs.append(dict(r))
        return self._fb_future(op, refs)

    async def _fb_async(self, op: str, data: Iterable[Datum], loss_fn: LossFn) -> APIFuture[ForwardOutput]:
        chunks = self._chunks(data, loss_fn)
        async with self._async_lock():
            refs = []
            for body in chunks:
                seq = self._seq
                r = await self._t.arequest(
                    "POST", f"/v1/runs/{self.id}/{op}", json={**body, "seq_id": seq}, method_name=f"run.{op}"
                )
                self._seq = seq + 1
                refs.append(dict(r))
        return self._fb_future(op, refs)

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
        return {
            "name": name,
            "include_optimizer": include_optimizer,
            "ttl_seconds": ttl_seconds,
            "user_metadata": user_metadata or {},
        }


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
