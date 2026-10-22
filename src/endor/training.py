"""Training custom models: datasets, runs and evaluations, all at the organization's level.

    client.datasets.upload("train", rows)
    with client.runs.create("decider-2b", model="router-2") as run:    # or client.custom_models.create("router-2", ...)
        ...                                                             # forward_backward, optim_step
        run.save_checkpoint().result()                                  # saved as the custom model "router-2"
    client.evaluations.create("router-2", "heldout").result()          # in production or archived
    client.custom_models.promote("router-2")                            # now model="router-2" answers

A save makes a custom model, which starts archived: it can be downloaded and evaluated, and decisions can call it once
it is promoted (``client.custom_models``).
"""

from __future__ import annotations

import builtins
import math
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any

from . import _constants as C
from . import _predict
from . import data as D
from ._http import Transport
from ._log import logger
from ._pages import list_all
from ._parse import parse
from .futures import APIFuture
from .runs import Run, check_model_name
from .types import DatasetInfo, DecisionRow, Evaluation, LoraConfig

__all__ = ["Datasets", "Runs", "Evaluations", "MAX_UPLOAD_ROWS", "git_identity"]

MAX_UPLOAD_ROWS = C.MAX_UPLOAD_ROWS

WAIT_NOTICE_S = 10.0
"""``runs.create(wait=True, progress=True)`` prints its line to stderr once the wait for a GPU passes this long."""


class Datasets:
    """``client.datasets``: labeled decision rows, for training and evaluating custom models."""

    def __init__(self, transport: Transport) -> None:
        self._t, self._base = transport, "/v1/datasets"

    def upload(self, name: str, rows: Iterable[Any]) -> DatasetInfo:
        """Upload decision rows (README, "Data format"). Rows may be dicts or ``DecisionRow`` objects; the other
        supported row formats are converted. At most 50,000 rows per dataset. Datasets are immutable: to change one,
        upload under a new name. A name the organization already has raises ``ConflictError``; to run a pipeline
        again, use the dataset that is there (``get(name)``) or catch the error. States are text or JSON: the API
        refuses images in training data (422)."""
        wire = [D.to_row(r).to_wire() for r in rows]
        if not wire:
            raise ValueError("no rows to upload")
        if len(wire) > MAX_UPLOAD_ROWS:
            raise ValueError(f"at most {MAX_UPLOAD_ROWS} rows per dataset (got {len(wire)}); split the data")
        r = self._t.request(
            "POST", self._base, json={"name": name, "rows": wire}, method_name="datasets.upload", idempotent=True
        )
        return parse(DatasetInfo, r, f"POST {self._base}")

    def list(self) -> builtins.list[DatasetInfo]:
        """Every dataset, newest first."""
        return [parse(DatasetInfo, d, f"GET {self._base}") for d in list_all(self._t, self._base, {}, "datasets.list")]

    def get(self, name: str) -> DatasetInfo:
        """One dataset's counts; ``NotFoundError`` if missing."""
        path = f"{self._base}/{name}"
        return parse(DatasetInfo, self._t.request("GET", path, method_name="datasets.get"), f"GET {path}")

    def rows(self, name: str, page: int = 500) -> Iterator[DecisionRow]:
        """Iterate over a dataset's rows, fetching ``page`` rows per request."""
        offset, path = 0, f"{self._base}/{name}/rows"
        while True:
            r = self._t.request("GET", path, params={"limit": page, "offset": offset}, method_name="datasets.rows")
            items = r["items"]
            yield from (parse(DecisionRow, x, f"GET {path}") for x in items)
            offset += len(items)
            if not items or offset >= r.get("total", offset):
                return

    def delete(self, name: str) -> None:
        """Delete a dataset. Evaluations made on it keep their results."""
        self._t.request("DELETE", f"{self._base}/{name}", method_name="datasets.delete")


class Runs:
    """``client.runs``: training runs for custom models."""

    def __init__(self, transport: Transport, capture: bool) -> None:
        self._t, self._capture = transport, capture

    def create(
        self,
        base_model: str | None = None,
        *,
        model: str | None = None,
        rank: int | None = None,
        alpha: float | None = None,
        seed: int | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_readout: bool | None = None,
        from_model: str | None = None,
        include_optimizer: bool = False,
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        user_metadata: dict[str, Any] | None = None,
        total_steps: int | None = None,
        wandb: bool | None = None,
        wait: bool = True,
        progress: bool = True,
    ) -> Run:
        """Start a run: a fresh adapter on ``base_model`` (an id from ``client.base_models()``), or one warm-started
        from ``from_model`` (a custom model's name, which sets the base model: another ``base_model`` is a 422, an
        unknown name ``NotFoundError``, 404 ``unknown_model``; with ``include_optimizer=True`` training resumes
        exactly).

        ``model``: the custom model name ``run.save_checkpoint()`` saves under when it is given none. It must be free
        (``ConflictError`` here if a custom model or project has it). ``name`` is the run's own label.

        LoRA settings left as None default to rank 16, alpha 32, attention and MLP, no readout for a fresh run. With
        ``from_model`` they are inherited from the saved model, and only the ones you pass are sent (a value that
        conflicts with the model is a 422).

        Provisioning a trainer takes under a minute on a warm GPU; starting a cold one can take a few minutes. With
        ``wait=True`` this blocks until the run is ready and logs progress at INFO on the ``endor`` logger; with
        ``progress=True`` (the default) it also prints one line to stderr once the wait passes 10 seconds
        (``run run_...: waiting for a GPU (starting a cold one can take a few minutes)…``), and ``progress=False``
        keeps stderr quiet. If the wait is interrupted (Ctrl-C, an error) the run is closed. With ``wait=False`` it
        returns at once and ``run.ready`` is the future.

        Close every run you create (``with client.runs.create(...) as run:``): runs that aren't closed, idle ones
        included, count toward the org's open-run limit (25 by default; ``max_active_runs`` in
        ``client.whoami().limits``, which ``endor whoami`` shows, is the org's own), and one more raises
        ``LimitReachedError``. A run without calls is parked (``idle``)
        after 2 minutes. Until you close it, the run sends a heartbeat every minute from a background thread, so it
        stays open however long you wait; if your process ends without closing it, Endor closes it an hour later.
        ``config`` is free-form and shown on the dashboard; when the client was created with ``capture=True`` the SDK
        adds the LoRA settings and the current git commit (never file contents).

        ``total_steps``: the optimizer steps you plan, so the dashboard shows a progress bar and an ETA (change it
        later with ``run.set_total_steps``). ``wandb``: log this run to Weights & Biases (True or False); None logs
        it when the organization has a W&B connection (dashboard: Settings > Integrations). ``run.info().wandb_url``
        links to the W&B run.
        """
        if base_model is None and from_model is None:
            raise ValueError('pass the base model to train on, e.g. client.runs.create("decider-2b"), or from_model=')
        given = {
            "rank": rank,
            "alpha": alpha,
            "seed": seed,
            "train_attn": train_attn,
            "train_mlp": train_mlp,
            "train_readout": train_readout,
        }
        given = {k: v for k, v in given.items() if v is not None}
        # A fresh run gets the defaults (rank 16, alpha 32, attention and MLP); a run from a saved model inherits
        # that model's settings, so only what the caller set explicitly is sent.
        lora = given if from_model else LoraConfig(**given).model_dump()
        body: dict[str, Any] = {
            "base_model": base_model,
            "from_model": from_model,
            "model": model,
            "include_optimizer": include_optimizer,
            "lora": lora,
            "name": name,
            "tags": tags or [],
            "config": dict(config or {}),
            "user_metadata": user_metadata or {},
        }
        if model is not None:
            check_model_name(model)
        if total_steps is not None:
            if total_steps < 1:
                raise ValueError("total_steps must be at least 1 (or None)")
            body["total_steps"] = total_steps
        if wandb is not None:
            body["wandb"] = wandb
        if self._capture:
            body["config"] = {
                **body["config"],
                "base_model": base_model,
                "from_model": from_model,
                "lora": lora,
            }
            body["code_hash"] = git_identity()
        r = self._t.request("POST", "/v1/runs", json=body, method_name="runs.create", idempotent=True)
        run = Run(self._t, r, endpoint="POST /v1/runs")
        run._start_heartbeat()
        if wait:
            try:
                _wait_ready(run, progress=progress)
            except BaseException:
                logger.warning("run %s: stopped waiting for provisioning; closing it so its GPU is released", run.id)
                run._close_on_exit(failing=True)
                raise
        return run

    def get(self, run_id: str) -> Run:
        """A handle on an existing run, to continue it: its sequence numbers continue from the server's. ``run.info_``
        is the run as fetched, e.g. ``run.info_.step`` to resume from the right step; ``run.info()`` fetches it again.

        Like a run from ``create``, the handle keeps an open run open: it sends a heartbeat every minute from a
        background thread until the run is closed (``run.close()``, a ``with`` block, or elsewhere) or the handle is
        garbage collected."""
        path = f"/v1/runs/{run_id}"
        run = Run(self._t, self._t.request("GET", path, method_name="runs.get"), endpoint=f"GET {path}")
        run._start_heartbeat()
        return run

    def list(self, tag: str | None = None, limit: int | None = None, offset: int = 0) -> builtins.list[Run]:
        """Your runs, newest first, optionally those carrying ``tag``: all of them, or at most ``limit``. These handles
        are for looking: they send no heartbeat unless used in a ``with`` block (``runs.get(id)`` re-attaches to one
        to continue it)."""
        items = list_all(self._t, "/v1/runs", {"tag": tag}, "runs.list", limit, offset)
        return [Run(self._t, x, endpoint="GET /v1/runs") for x in items]


class Evaluations:
    """``client.evaluations``: how good a custom model (in production or archived) or a base model is on decision
    rows."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def create(
        self,
        model: str,
        dataset: str | None = None,
        run_id: str | None = None,
        *,
        rows: Iterable[Any] | None = None,
    ) -> APIFuture[Evaluation]:
        """Score a model server-side on a dataset, or on ``rows`` sent with the call (decision rows in any shape
        ``client.datasets.upload`` takes, at most 50,000 and 32 MB; nothing is stored as a dataset). One of the two.
        ``model`` is a custom model's name (in production or archived) or a base model id (the untuned base, as a
        baseline).

        A row with labels is scored on its labeled questions; a row without labels is answered on every question and
        not scored (``client.predict`` returns those answers). The result's ``results`` has ``n`` (the questions
        scored), then ``overall``, ``by_type`` and ``by_question``, each with accuracy, NLL, Brier, ECE, mean
        confidence and ``selective`` (``{"0.5": {"accuracy", "coverage"}, ...}`` for the thresholds 0.5, 0.7, 0.9 and
        0.95), and ``rows_url``, a link to every answer (one JSON line per question asked) that expires after 1 hour.
        While it runs, ``rows_done`` of ``rows_total`` shows how far it is. With ``run_id``, the evaluation appears on
        that run's page. Accuracies are fractions from 0 to 1.

        Raises:
            ValueError: Neither a dataset nor rows, or both.
            NotFoundError: An unknown model (404 ``unknown_model``) or dataset.
            UnprocessableEntityError: A bad row (``param``: ``rows.<i>...``) or a question the model can't read.
        """
        body = _predict.evaluation_body(model, dataset, rows, run_id)
        r = self._t.request("POST", "/v1/evaluations", json=body, method_name="evaluations.create", idempotent=True)

        def fetch(x: dict[str, Any]) -> Evaluation:
            return self.get(str(x["evaluation_id"]))

        return APIFuture(self._t, r["future_id"], fetch, method="evaluations.create")

    def get(self, evaluation_id: str) -> Evaluation:
        """One evaluation, with ``rows_done`` of ``rows_total`` while it runs."""
        path = f"/v1/evaluations/{evaluation_id}"
        return parse(Evaluation, self._t.request("GET", path, method_name="evaluations.get"), f"GET {path}")

    def list(self, run_id: str | None = None, model: str | None = None) -> builtins.list[Evaluation]:
        """Server-side and client-recorded evaluations, newest first (all pages), optionally of one run or model."""
        items = list_all(self._t, "/v1/evaluations", {"run_id": run_id, "model": model}, "evaluations.list")
        return [parse(Evaluation, e, "GET /v1/evaluations") for e in items]


def _wait_ready(run: Run, log_every: float = 30.0, progress: bool = True) -> None:
    """Block until the run's trainer is ready, logging at INFO every ``log_every`` seconds; with ``progress``, also
    print one line to stderr once the wait passes ``WAIT_NOTICE_S`` (a cold GPU takes minutes, and Python hides INFO
    logs by default)."""
    started = time.monotonic()
    if run.info_.status in ("ready", "idle", "closed", "failed"):
        return
    logger.info("run %s: provisioning a trainer for %s ...", run.id, run.info_.base_model)
    next_log, notice_at = log_every, WAIT_NOTICE_S if progress else math.inf
    while True:
        try:
            run.ready.result(timeout=max(0.0, min(next_log, notice_at) - (time.monotonic() - started)))
            break
        except TimeoutError:
            waited = time.monotonic() - started
            if waited >= notice_at:
                notice_at = math.inf
                print(
                    f"run {run.id}: waiting for a GPU (starting a cold one can take a few minutes)…",
                    file=sys.stderr,
                    flush=True,
                )
            if waited >= next_log:
                next_log = waited + log_every
                logger.info("run %s: still provisioning (%.0fs)", run.id, waited)
    logger.info("run %s: ready after %.0fs", run.id, time.monotonic() - started)
    run.info()


def git_identity() -> str | None:
    """``"<commit>"`` or ``"<commit>+dirty"`` for the working directory's git repository; None outside git.
    Only the hash and the dirty flag are ever sent, never file contents."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2).stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2).stdout
        return sha + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None
