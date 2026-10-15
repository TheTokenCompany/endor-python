"""Run: one adapter on one base model inside a project, driven step by step.

    run = project.runs.create(rank=16)               # on the project's first base model
    fb = run.forward_backward(datums)                 # gradients accumulate
    opt = run.optim_step(learning_rate=1e-4)          # AdamW step, then zero gradients
    fb.result(); opt.result()                         # submit both, then wait: no idle round trip
    model = run.save_checkpoint("v1").result()        # "tickets/v1", usable with client.system_one at once

In a loop, ``endor.Steps`` sends each step before the last one's results are read (see its docstring).

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
import weakref
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import TypeAdapter

from . import _constants as C
from . import data as D
from ._http import Transport
from ._log import logger
from .errors import ConflictError, NotFoundError
from .futures import APIFuture
from .types import (
    AdamParams,
    Answer,
    Datum,
    Evaluation,
    ForwardOutput,
    LossFn,
    MetricPoint,
    OptimStepOutput,
    RunInfo,
    option_keys,
    question_dict,
)

__all__ = ["Run", "MAX_DATUMS_PER_CALL"]

_FINAL = ("closed", "failed")
_MODEL_NAME = re.compile(C.SDK_MODEL_NAME)
_CLOSE_POLL_S = 1.0

MAX_DATUMS_PER_CALL = C.MAX_DATUMS_PER_CALL
"""Bigger batches are split into several requests; the merged result and the gradients are the same."""

MAX_PREDICT_QUESTIONS = C.MAX_PREDICT_QUESTIONS_PER_CALL
"""``predict`` sends at most this many questions per request (across rows); longer lists are split, and the answers
come back merged in row order."""


class Run:
    """A training run. Get one from ``project.runs.create(...)``, ``project.runs.get(id)`` or
    ``project.runs.list()``.

    Training is billed for the GPU time its calls use. A run without calls is parked (``idle``) after 2 minutes, and an
    org can have at most 5 runs that aren't closed (idle ones included). While a run is open (from ``runs.create``, or
    in a ``with`` block) a background thread sends a heartbeat every minute, so it stays open however long you wait
    between calls; a run whose process ended without closing it is closed 5 minutes later. Use it as a context
    manager so it is closed even on errors::
    Training is billed for the GPU time its calls use. A run without calls is parked (``idle``) after 2 minutes and
    closed after an hour, and an org can have at most 25 runs that aren't closed (idle ones included). Use it as a
    context manager so it is closed even on errors::

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
        self._heartbeat: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, project={self.project!r}, base_model={self.info_.base_model!r})"

    def __enter__(self) -> Run:
        self._start_heartbeat()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._close_on_exit(exc is not None)

    async def __aenter__(self) -> Run:
        self._start_heartbeat()
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

    def _start_heartbeat(self) -> None:
        """Tell the API every ``RUN_HEARTBEAT_S`` that this client is still there, from a daemon thread, until the run
        is closed (here or elsewhere) or this handle is garbage collected. Once per handle."""
        if self._heartbeat is not None or self._stop_heartbeat.is_set() or self.info_.status in _FINAL:
            return
        weakref.finalize(self, self._stop_heartbeat.set)  # a handle nobody holds stops, so the API closes its run
        self._heartbeat = threading.Thread(
            target=_beat, args=(self._t, self.id, self._stop_heartbeat), name=f"endor-heartbeat-{self.id}", daemon=True
        )
        self._heartbeat.start()

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

    def predict(self, rows: Iterable[Any]) -> APIFuture[list[dict[str, Answer]]]:
        """Answer decision rows with the run's current adapter, without saving a model: one ``{name: answer}`` per
        row, in order, as ``client.system_one`` answers them.

        Use it to score a candidate before you save it. The answers match a decision on the model saved at this
        point within the usual train/serve tolerance (bf16 rounding from the batch shape, up to a few hundredths in a
        probability). ``rows`` are decision rows in any shape ``endor.data.to_row`` takes; labels, ids and weights
        are ignored. No gradients, and the run's training state doesn't change. It runs in order with the run's
        other calls (after the steps sent before it) and is billed as training GPU time, like ``forward``. Lists
        over ``MAX_PREDICT_QUESTIONS`` questions are split into several requests; every row is checked before any
        is sent. Text and JSON states only (no images yet).
        """
        return self._predict_future(self._send("predict", self._predict_chunks(rows)))

    async def predict_async(self, rows: Iterable[Any]) -> APIFuture[list[dict[str, Answer]]]:
        """``predict`` for async code: submits without blocking; ``await`` the returned future for the answers."""
        return self._predict_future(await self._send_async("predict", self._predict_chunks(rows)))

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

        ``name``: lowercase letters, digits, ``.``, ``_`` and ``-``, up to 63 characters, starting with a letter or
        digit, and not ``base`` (the project's base model).
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
        self._stop_heartbeat.set()
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
        self._stop_heartbeat.set()
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
        """Refresh and return the run's status, ``step`` and ``next_seq_id``, with its progress: ``total_steps``,
        ``seconds_per_step`` (the median over the last 20 steps), ``progress`` (0 to 1) and ``eta_seconds``, and its
        Weights & Biases link (``wandb_url``) once Endor has logged to it."""
        self.info_ = RunInfo.model_validate(self._t.request("GET", f"/v1/runs/{self.id}", method_name="run.info"))
        return self.info_

    def set_total_steps(self, total_steps: int | None) -> RunInfo:
        """Tell Endor how many optimizer steps you plan (None: unknown), so the dashboard shows a progress bar and an
        ETA for this run. ``runs.create(total_steps=...)`` sets it at the start; the recipes set it for you."""
        if total_steps is not None and total_steps < 1:
            raise ValueError("total_steps must be at least 1 (or None)")
        self.info_ = RunInfo.model_validate(
            self._t.request(
                "PATCH", f"/v1/runs/{self.id}", json={"total_steps": total_steps}, method_name="run.set_total_steps"
            )
        )
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
        return self._fb_future(op, self._send(op, self._chunks(data, loss_fn)))

    async def _fb_async(self, op: str, data: Iterable[Datum], loss_fn: LossFn) -> APIFuture[ForwardOutput]:
        return self._fb_future(op, await self._send_async(op, self._chunks(data, loss_fn)))

    def _send(self, op: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """POST the chunks of one call with consecutive sequence numbers; their future refs, in order."""
        with self._lock:
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
        return refs

    async def _send_async(self, op: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        return refs

    def _cancel_partial(self, op: str, refs: list[dict[str, Any]], total: int) -> None:
        """A later chunk was refused by the server (402, 429, ...): cancel the chunks it already accepted, so their
        gradients don't reach the next optim_step (and a half-sent predict doesn't run). A chunk that already started
        can't be cancelled; say so."""
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
                "so some of them may still run%s",
                self.id,
                op,
                len(refs),
                total,
                states,
                " and their gradients be applied by the next optim_step" if op == "forward_backward" else "",
            )

    def _predict_future(self, refs: list[dict[str, Any]]) -> APIFuture[list[dict[str, Answer]]]:
        futures = [APIFuture(self._t, r["future_id"], _parse_answers, method="run.predict") for r in refs]
        if len(futures) == 1:
            return futures[0]
        return APIFuture(self._t, children=list(futures), combine=_concat_answers, method="run.predict")

    @staticmethod
    def _predict_chunks(rows: Iterable[Any]) -> list[dict[str, Any]]:
        """Each row's state and questions in wire form, in requests of at most ``MAX_PREDICT_QUESTIONS`` questions.
        Every row is checked before any request is sent, so a bad row never leaves half a call sent."""
        wire: list[dict[str, Any]] = []
        for i, obj in enumerate(rows):
            try:
                row = D.to_row(obj)
            except ValueError as e:
                raise ValueError(f"rows[{i}]: {e}") from e
            if len(row.questions) > C.MAX_QUESTIONS_PER_REQUEST:
                raise ValueError(
                    f"rows[{i}]: at most {C.MAX_QUESTIONS_PER_REQUEST} questions per row, got {len(row.questions)}"
                )
            wire.append({"state": row.state, "questions": {n: question_dict(q) for n, q in row.questions.items()}})
        if not wire:
            raise ValueError("no rows: pass at least one decision row")
        chunks: list[list[dict[str, Any]]] = [[]]
        size = 0
        for w in wire:
            if size and size + len(w["questions"]) > MAX_PREDICT_QUESTIONS:
                chunks.append([])
                size = 0
            chunks[-1].append(w)
            size += len(w["questions"])
        return [{"rows": c} for c in chunks]

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


def _beat(t: Transport, run_id: str, stop: threading.Event) -> None:
    """The heartbeat thread: it holds the transport, never the ``Run``. A failure is logged at debug and the next beat
    tries again; a run the API reports closed (409) or gone (404) stops it."""
    while not stop.wait(C.RUN_HEARTBEAT_S):
        try:
            t.request("POST", f"/v1/runs/{run_id}/heartbeat", method_name="run.heartbeat", remember=False)
        except (ConflictError, NotFoundError):
            logger.debug("run %s: heartbeat stopped: the run is closed or gone", run_id, exc_info=True)
            return
        except Exception:
            logger.debug("run %s: heartbeat failed", run_id, exc_info=True)


def check_model_name(name: str) -> None:
    """The API's rule for model names saved from the SDK; raises ValueError naming the problem."""
    if name == C.BASE_MODEL_NAME:
        raise ValueError('"base" is reserved (base models are called "<project>/<base id>"); pick another model name')
    if not isinstance(name, str) or not _MODEL_NAME.fullmatch(name):
        raise ValueError(
            f"bad model name {name!r}: lowercase letters, digits, '.', '_' or '-', starting with a letter or digit, "
            "63 characters at most"
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


_ANSWERS: TypeAdapter[list[dict[str, Answer]]] = TypeAdapter(list[dict[str, Answer]])


def _parse_answers(result: Any) -> list[dict[str, Answer]]:
    """A predict future's result: per row, in order, its answers by question name."""
    return _ANSWERS.validate_python(result["answers"])


def _concat_answers(parts: list[list[dict[str, Answer]]]) -> list[dict[str, Answer]]:
    """The answers of a predict sent in several requests, in row order."""
    return [a for part in parts for a in part]
