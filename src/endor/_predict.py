"""Server-side evaluations on rows sent inline, and predictions (``client.predict``) built on them.

An evaluation can take its rows in the create call instead of a dataset's name (``POST /v1/evaluations`` with
``rows``). A row with labels is scored on its labeled questions; a row
without labels is answered on every question and not scored. A prediction is such an evaluation of rows sent without
their labels: the SDK waits for its future (printing a progress line every PROGRESS_EVERY_S while it runs), fetches
its results file (``results["rows_url"]``: one JSON line per question asked, with the row's ``index`` and the
``answer`` as ``system_one`` gives it) and returns every row's answers, in the rows' order.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from pydantic import TypeAdapter

from . import _constants as C
from . import data as D
from ._download import fetch_bytes, fetch_bytes_async
from ._http import Transport
from ._parse import parse
from .errors import EndorError
from .futures import APIFuture
from .types import Answer, Evaluation

PROGRESS_EVERY_S = 10.0
"""How often a prediction checks and prints its evaluation's progress while it runs."""
_ANSWER: TypeAdapter[Answer] = TypeAdapter(Answer)
Answers = dict[str, Answer]


def wire_rows(rows: Iterable[Any], *, labels: bool = True) -> list[dict[str, Any]]:
    """Rows in any shape ``endor.data.to_row`` takes, as the API takes them (at most MAX_UPLOAD_ROWS); without their
    labels when ``labels`` is false."""
    wire = [D.to_row(r).to_wire() for r in rows]
    if not wire:
        raise ValueError("no rows")
    if len(wire) > C.MAX_UPLOAD_ROWS:
        raise ValueError(f"at most {C.MAX_UPLOAD_ROWS} rows per call (got {len(wire)}); split the rows")
    if not labels:
        for w in wire:
            w.pop("labels", None)
    return wire


def evaluation_body(model: str, dataset: str | None, rows: Iterable[Any] | None, run_id: str | None) -> dict[str, Any]:
    """The create call's body: a dataset's name or the rows, one of the two."""
    if (dataset is None) == (rows is None):
        raise ValueError("pass a dataset name or rows=..., one of the two")
    body: dict[str, Any] = {"model": model}
    if dataset is not None:
        body["dataset"] = dataset
    body["training_run_id"] = run_id
    if rows is not None:
        body["rows"] = wire_rows(rows)
    return body


def prediction(model: str, rows: Iterable[Any]) -> tuple[dict[str, Any], int]:
    """(body, number of rows) of a prediction: its rows without labels, for a custom model or a base model."""
    if not model:
        raise ValueError("model is a custom model's name or a base model id")
    wire = wire_rows(rows, labels=False)
    return {"model": model, "rows": wire}, len(wire)


def wait(t: Transport, created: dict[str, Any], progress: bool, timeout: float | None, method: str) -> Evaluation:
    """Wait for a created evaluation (``{"future_id", "evaluation_id"}``) and return it, printing its progress to
    stderr every PROGRESS_EVERY_S while it runs when ``progress``. ``TimeoutError`` once ``timeout`` seconds have
    passed (it keeps running); ``OperationFailedError`` when it fails."""
    path, deadline = f"/v1/evaluations/{created['evaluation_id']}", _deadline(timeout)
    future: APIFuture[Any] = APIFuture(t, created["future_id"], method=method)
    while True:
        step, last = _step(deadline)
        try:
            future.result(timeout=step)
            break
        except TimeoutError:
            if last:
                raise
        if progress:
            _show(parse(Evaluation, t.request("GET", path, method_name=method), f"GET {path}"))
    return parse(Evaluation, t.request("GET", path, method_name=method), f"GET {path}")


async def wait_async(
    t: Transport, created: dict[str, Any], progress: bool, timeout: float | None, method: str
) -> Evaluation:
    """``wait`` for async code."""
    path, deadline = f"/v1/evaluations/{created['evaluation_id']}", _deadline(timeout)
    future: APIFuture[Any] = APIFuture(t, created["future_id"], method=method)
    while True:
        step, last = _step(deadline)
        try:
            await future.result_async(timeout=step)
            break
        except TimeoutError:
            if last:
                raise
        if progress:
            _show(parse(Evaluation, await t.arequest("GET", path, method_name=method), f"GET {path}"))
    return parse(Evaluation, await t.arequest("GET", path, method_name=method), f"GET {path}")


def answers(t: Transport, ev: Evaluation, n: int) -> list[Answers]:
    """Every row's answers from a completed evaluation's results file, in the rows' order."""
    return _read_answers(fetch_bytes(_rows_url(ev), "the evaluation's results", transport=t._transport), n, ev.id)


async def answers_async(t: Transport, ev: Evaluation, n: int) -> list[Answers]:
    """``answers`` for async code."""
    blob = await fetch_bytes_async(_rows_url(ev), "the evaluation's results", transport=t._atransport)
    return _read_answers(blob, n, ev.id)


def progress_line(ev: Evaluation) -> str | None:
    """``evaluation evl_...: 1,200/10,000 rows (12%) · ~2 min left``, or None before it knows its rows."""
    done, total = ev.rows_done, ev.rows_total
    if done is None or not total:
        return None
    eta = ""
    if ev.started_at is not None and 0 < done < total:
        left = (total - done) * (datetime.now(timezone.utc) - ev.started_at).total_seconds() / done
        eta = f" · ~{_duration(left)} left"
    return f"evaluation {ev.id}: {done:,}/{total:,} rows ({done / total:.0%}){eta}"


def _show(ev: Evaluation) -> None:
    line = progress_line(ev)
    if line:
        print(line, file=sys.stderr, flush=True)


def _deadline(timeout: float | None) -> float | None:
    return None if timeout is None else time.monotonic() + timeout


def _step(deadline: float | None) -> tuple[float, bool]:
    """How long the next wait is, and whether it is the last (it ends at the deadline)."""
    if deadline is None:
        return PROGRESS_EVERY_S, False
    left = max(0.0, deadline - time.monotonic())
    return min(PROGRESS_EVERY_S, left), left <= PROGRESS_EVERY_S


def _rows_url(ev: Evaluation) -> str:
    url = (ev.results or {}).get("rows_url")
    if not isinstance(url, str):
        raise EndorError(f"evaluation {ev.id} has no results file (status {ev.status})")
    return url


def _read_answers(blob: bytes, n: int, evaluation_id: str) -> list[Answers]:
    out: list[Answers] = [{} for _ in range(n)]
    for line in blob.splitlines():
        if line.strip():
            x = json.loads(line)
            out[x["index"]][x["question"]] = parse(_ANSWER, x["answer"], f"evaluation {evaluation_id}'s results")
    return out


def _duration(seconds: float) -> str:
    """A duration as "40s", "12 min" or "2 h 5 min" (as the training recipe's progress line)."""
    if seconds < 60:
        return f"{max(1, round(seconds))}s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min" if minutes % 60 else f"{minutes // 60} h"
