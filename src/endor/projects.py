"""Projects: the unit of work. A project, keyed by its name, holds base models, datasets, runs, models and evaluations.

project = client.projects.get_or_create("tickets", base_models=["decider-2b"])   # kind="custom"
project.add_base_model("gev-26b")          # "tickets/gev-26b" answers at once
project.datasets.upload("train", rows)
run = project.runs.create()                # on the project's first base model, decider-2b
project.models.list()                      # "tickets/decider-2b", "tickets/gev-26b", then every saved model
project.evaluate("decider-2b", "heldout").result()   # names resolve in the project: "tickets/decider-2b"
"""

from __future__ import annotations

import builtins
import os
import subprocess
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal

from . import _constants as C
from . import _predict
from . import data as D
from ._http import Transport
from ._images import Images
from ._log import logger
from .errors import ConflictError, NotFoundError
from .futures import APIFuture
from .runs import Run
from .types import (
    DatasetInfo,
    DecisionRow,
    DownloadedFile,
    Evaluation,
    Learning,
    LoraConfig,
    ModelInfo,
    ProjectInfo,
    WandbSettings,
)

ProjectKind = Literal["custom", "managed"]
"""``managed``: continuous learning, the recommended way. Endor trains new versions of the project's model from its
decisions and your corrections (``client.corrections``). ``custom``: you train models with the SDK."""


__all__ = ["Projects", "Project", "Datasets", "Runs", "Models", "MAX_UPLOAD_ROWS"]

MAX_UPLOAD_ROWS = C.MAX_UPLOAD_ROWS


class Projects:
    """``client.projects``: create, fetch and list projects."""

    def __init__(self, transport: Transport, capture: bool, images: Images | None = None) -> None:
        self._t, self._capture = transport, capture
        self._images = images or Images(transport)

    def create(
        self,
        name: str,
        description: str | None = None,
        *,
        kind: ProjectKind = "custom",
        base_models: Sequence[str] | None = None,
        **removed: Any,
    ) -> Project:
        """A new project. Names are lowercase, ``[a-z0-9._-]``, up to 63 characters, and permanent (they are part
        of every model id). Raises ``ConflictError`` if you already have one with that name, and
        ``LimitReachedError`` (a ``ConflictError``) when the org has as many projects as it may.

        ``kind`` is ``"managed"`` (continuous learning: Endor trains new versions from the project's decisions and
        your corrections; its decisions cost 50% more) or ``"custom"`` (you train models with the SDK; the default
        here). It never changes.

        ``base_models``: base model ids from ``client.base_models()``, for example ``["decider-2b"]``. Each answers at
        once as ``"<project>/<base id>"``, and ``"<project>"`` answers with the first. A continuous learning project
        has exactly one, chosen here for good (``ValueError`` otherwise); a custom project may start with none (a run
        on a base model adds it) and adds and removes them later with ``project.add_base_model`` and
        ``project.remove_base_model``."""
        bases = _base_list(base_models, removed, "projects.create")
        if kind == "managed" and len(bases) != 1:
            raise ValueError(
                "a continuous learning project has exactly one base model, chosen at creation: "
                'base_models=["decider-2b"]'
            )
        body: dict[str, Any] = {"name": name, "description": description, "kind": kind, "base_models": bases}
        r = self._t.request("POST", "/v1/projects", json=body, method_name="projects.create", idempotent=True)
        return Project(self._t, r, self._capture)

    def get(self, name: str) -> Project:
        """Your project called ``name``; ``NotFoundError`` if there is none."""
        return Project(
            self._t, self._t.request("GET", f"/v1/projects/{name}", method_name="projects.get"), self._capture
        )

    def get_or_create(
        self,
        name: str,
        description: str | None = None,
        *,
        kind: ProjectKind = "custom",
        base_models: Sequence[str] | None = None,
        **removed: Any,
    ) -> Project:
        """The project called ``name``, created if missing (with these settings). Safe to call at the top of every
        script. An existing custom project gets the ``base_models`` it doesn't have yet, and keeps its other base models
        and its description. Raises ``ValueError`` if the existing project has another ``kind`` (it can't change, so
        use another project name), or for an existing continuous learning project with another base model (its base is
        chosen once)."""
        bases = _base_list(base_models, removed, "projects.get_or_create")
        try:
            p = self.get(name)
        except NotFoundError:
            return self.create(name, description, kind=kind, base_models=bases)
        if p.info_.kind != kind:
            raise ValueError(
                f"project {name!r} exists as a {p.info_.kind} project, not a {kind} project; a project's kind can't "
                "change, so pick another name"
            )
        if kind == "managed":
            if bases and bases != p.info_.base_models:
                raise ValueError(
                    f"continuous learning project {name!r} learns on {p.info_.base_models}, not {bases}; its base "
                    "model is chosen once, so pick another name"
                )
            return p
        for b in bases:
            if b not in p.info_.base_models:
                try:
                    p.add_base_model(b)
                except ConflictError as e:  # added meanwhile, by another process
                    if e.code != "conflict":
                        raise
                    p.info()
        return p

    def list(self, limit: int | None = None, offset: int = 0) -> list[Project]:
        """Your projects, newest first: all of them, or at most ``limit`` starting at ``offset``."""
        items = _list_all(self._t, "/v1/projects", {}, "projects.list", limit, offset)
        return [Project(self._t, p, self._capture) for p in items]


class Project:
    """One project. ``.datasets``, ``.runs`` and ``.models`` hold what it contains."""

    def __init__(
        self, transport: Transport, info: dict[str, Any], capture: bool = True, images: Images | None = None
    ) -> None:
        self._t = transport
        self._images = images or Images(transport)
        self.info_ = ProjectInfo.model_validate(info)
        """Name, description and counts as last fetched."""
        self.name: str = self.info_.name
        self.datasets = Datasets(transport, self.name, self._images)
        self.runs = Runs(transport, self.name, capture, self._images)
        self.models = Models(transport, self.name)

    def __repr__(self) -> str:
        return f"Project({self.name!r})"

    def info(self) -> ProjectInfo:
        """Refresh the project's counts."""
        self.info_ = ProjectInfo.model_validate(
            self._t.request("GET", f"/v1/projects/{self.name}", method_name="project.info")
        )
        return self.info_

    def learning(self) -> Learning:
        """A continuous learning project's learning, which is always on: its status, the way to the next training run
        (``progress``, 0 to 1), what ``"<project>"`` serves, the latest training runs (each kept or not, with its
        accuracy and loss) and whether recent traffic changed (``drift``).
        ``WrongProjectKindError`` for a custom project."""
        return Learning.model_validate(
            self._t.request("GET", f"/v1/projects/{self.name}/learning", method_name="project.learning")
        )

    def add_base_model(self, base_model: str) -> ProjectInfo:
        """Add a base model (an id from ``client.base_models()``) to the project, after the ones it has. It answers at
        once as ``"<project>/<base id>"``, and runs can train on it. Custom projects only: a continuous learning
        project keeps the one base model it was created with, and the API refuses another (``ConflictError`` or
        ``UnprocessableEntityError``). Raises ``ConflictError`` if the project has it, and ``UnprocessableEntityError``
        if the id is not a base model."""
        self.info_ = ProjectInfo.model_validate(
            self._t.request(
                "POST",
                f"/v1/projects/{self.name}/base_models",
                json={"base_model": base_model},
                method_name="project.add_base_model",
                idempotent=True,
            )
        )
        return self.info_

    def remove_base_model(self, base_model: str) -> ProjectInfo:
        """Take a base model off the project: ``"<project>/<base id>"`` stops answering, and ``"<project>"`` answers
        with the next base when this was the first. Saved models trained on it and open runs on it keep working, and
        a new run on it adds it again. Custom projects only: a continuous learning project keeps its one base model
        (``ConflictError``). Raises ``NotFoundError`` if the project doesn't have it."""
        self.info_ = ProjectInfo.model_validate(
            self._t.request(
                "DELETE",
                f"/v1/projects/{self.name}/base_models/{base_model}",
                method_name="project.remove_base_model",
            )
        )
        return self.info_

    def update(
        self,
        *,
        description: str | None = None,
        wandb: WandbSettings | dict[str, Any] | None = None,
    ) -> ProjectInfo:
        """Change the project's settings; arguments left as None are unchanged. The kind never changes; base models
        change with ``add_base_model`` and ``remove_base_model``.

        - ``description``: free text.
        - ``wandb``: ``{"enabled": True}``: new runs then log to Weights & Biases, from Endor's servers, in the W&B
          project ``endor-<project name>``. Needs the organization's W&B connection (dashboard: Settings >
          Integrations); ``ConflictError`` without it."""
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if wandb is not None:
            body["wandb"] = {"enabled": wandb.enabled} if isinstance(wandb, WandbSettings) else dict(wandb)
        if not body:
            raise ValueError("nothing to update: pass description or wandb")
        self.info_ = ProjectInfo.model_validate(
            self._t.request("PATCH", f"/v1/projects/{self.name}", json=body, method_name="project.update")
        )
        return self.info_

    def evaluate(
        self,
        model: str,
        dataset: str | None = None,
        run_id: str | None = None,
        *,
        rows: Iterable[Any] | None = None,
    ) -> APIFuture[Evaluation]:
        """Score a model server-side on one of this project's datasets, or on ``rows`` sent with the call (decision
        rows in any shape ``project.datasets.upload`` takes, at most 50,000 and 32 MB; nothing is stored as a
        dataset). One of the two. ``model`` is a base model id of this project (``"decider-2b"``), a saved model's
        name (``"v1"``), this project's name (a continuous learning project's newest version, else its first base
        model) or a full model id (``"<project>/<base id>"``, ``"<project>/<name>"``). The evaluation records the full
        id of the model that answered: ``"decider-2b"`` becomes ``"<project>/decider-2b"``.

        A row with labels is scored on its labeled questions; a row without labels is answered on every question and
        not scored (``client.predict`` returns those answers). The result's ``results`` has ``n`` (the questions
        scored), then ``overall``, ``by_type`` and ``by_question``, each with accuracy, NLL, Brier, ECE, mean
        confidence and ``selective`` (``{"0.5": {"accuracy", "coverage"}, ...}`` for the thresholds 0.5, 0.7, 0.9 and
        0.95), and ``rows_url``, a link to every answer (one JSON line per question asked) that expires after 1 hour.
        While it runs, ``rows_done`` of ``rows_total`` shows how far it is. With ``run_id``, the evaluation appears
        on that run's page.
        """
        body = _predict.evaluation_body(model, dataset, rows, run_id)
        r = self._t.request(
            "POST",
            f"/v1/projects/{self.name}/evaluations",
            json=body,
            method_name="project.evaluate",
            idempotent=True,
        )

        def fetch(x: dict[str, Any]) -> Evaluation:
            return Evaluation.model_validate(
                self._t.request("GET", f"/v1/evaluations/{x['evaluation_id']}", method_name="project.evaluate")
            )

        return APIFuture(self._t, r["future_id"], fetch, method="project.evaluate")

    def evaluations(self, run_id: str | None = None, model: str | None = None) -> list[Evaluation]:
        """Server-side and client-recorded evaluations in this project, newest first (all pages)."""
        params = {"run_id": run_id, "model": model}
        items = _list_all(self._t, f"/v1/projects/{self.name}/evaluations", params, "project.evaluations")
        return [Evaluation.model_validate(e) for e in items]

    def delete(self) -> None:
        """Delete the project with its datasets, models, runs and evaluations. Permanent. Close its runs first
        (``ConflictError`` with code ``run_active`` otherwise)."""
        self._t.request("DELETE", f"/v1/projects/{self.name}", method_name="project.delete")


class Datasets:
    """``project.datasets``: labeled decision rows, shared by the project's runs and evaluations."""

    def __init__(self, transport: Transport, project: str, images: Images | None = None) -> None:
        self._t, self._base = transport, f"/v1/projects/{project}/datasets"
        self._images = images or Images(transport)

    def upload(self, name: str, rows: Iterable[Any]) -> DatasetInfo:
        """Upload decision rows (README, "Data format"). Rows may be dicts or ``DecisionRow`` objects; the other
        supported row formats are converted. At most 50,000 rows per dataset. Datasets are immutable: to change
        one, upload under a new name. A name the project already has raises ``ConflictError``; to run a pipeline
        again, use the dataset that is there (``get(name)``) or catch the error. On a base model that reads images
        the states may hold ``endor.Image`` objects: each new image is uploaded once first (``client.images``) and
        the rows carry references (limited access: ``PermissionDeniedError`` until the org is approved)."""
        wire = [D.to_row(r).to_wire() for r in rows]
        if not wire:
            raise ValueError("no rows to upload")
        if len(wire) > MAX_UPLOAD_ROWS:
            raise ValueError(f"at most {MAX_UPLOAD_ROWS} rows per dataset (got {len(wire)}); split the data")
        self._images.prepare_states(wire)
        r = self._t.request(
            "POST",
            self._base,
            json={"name": name, "rows": wire},
            method_name="project.datasets.upload",
            idempotent=True,
        )
        return DatasetInfo.model_validate(r)

    def list(self) -> list[DatasetInfo]:
        """Every dataset in the project."""
        return [DatasetInfo.model_validate(d) for d in _list_all(self._t, self._base, {}, "project.datasets.list")]

    def get(self, name: str) -> DatasetInfo:
        """One dataset's counts; ``NotFoundError`` if missing."""
        return DatasetInfo.model_validate(
            self._t.request("GET", f"{self._base}/{name}", method_name="project.datasets.get")
        )

    def rows(self, name: str, page: int = 500) -> Iterator[DecisionRow]:
        """Iterate over a dataset's rows, fetching ``page`` rows per request."""
        offset = 0
        while True:
            r = self._t.request(
                "GET",
                f"{self._base}/{name}/rows",
                params={"limit": page, "offset": offset},
                method_name="project.datasets.rows",
            )
            items = r["items"]
            yield from (D.to_row(x) for x in items)
            offset += len(items)
            if not items or offset >= r.get("total", offset):
                return

    def delete(self, name: str) -> None:
        """Delete a dataset. Evaluations made on it keep their results."""
        self._t.request("DELETE", f"{self._base}/{name}", method_name="project.datasets.delete")


class Runs:
    """``project.runs``: training runs."""

    def __init__(self, transport: Transport, project: str, capture: bool, images: Images | None = None) -> None:
        self._t, self._project, self._capture = transport, project, capture
        self._images = images or Images(transport)

    def create(
        self,
        base_model: str | None = None,
        *,
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
    ) -> Run:
        """Start a run: a fresh adapter on ``base_model``, or one warm-started from ``from_model`` (a model of this
        project, as ``"name"`` or ``"<project>/name"``; with ``include_optimizer=True`` training resumes exactly).

        ``base_model`` left out: the project's first base model (``UnprocessableEntityError`` if it has none). A base
        model the project doesn't have yet is added to it. With ``from_model`` the run trains that model's base model,
        and another ``base_model`` is a 422.

        LoRA settings left as None default to rank 16, alpha 32, attention and MLP, no readout for a fresh run. With
        ``from_model`` they are inherited from the saved model, and only the ones you pass are sent (a value that
        conflicts with the model is a 422).

        Provisioning a trainer usually takes under a minute. With ``wait=True`` this blocks until the run is ready and
        logs progress on the ``endor`` logger; if the wait is interrupted (Ctrl-C, an error) the run is closed. With
        ``wait=False`` it returns at once and ``run.ready`` is the future.

        Close every run you create (``with project.runs.create(...) as run:``): an org can have at most 5 runs that
        aren't closed, idle ones included (a sixth raises ``LimitReachedError``). A run without calls is parked
        (``idle``) after 2 minutes. Until you close it, the run sends a heartbeat every minute from a background
        thread, so it stays open however long you wait; if your process ends without closing it, Endor closes it 5
        minutes later. Custom projects only: a continuous learning project raises ``WrongProjectKindError`` (Endor
        trains it). Close every run you create (``with project.runs.create(...) as run:``): an org can have at most 25
        runs that aren't closed, idle ones included (a 26th raises ``LimitReachedError``). A run without calls is parked
        (``idle``) after 2 minutes and closed after an hour. Custom projects only: a continuous learning project raises
        ``WrongProjectKindError`` (Endor trains it).
        ``config`` is free-form and shown on the dashboard; when the client was created with ``capture=True`` the
        SDK adds the LoRA settings and the current git commit (never file contents).

        ``total_steps``: the optimizer steps you plan, so the dashboard shows a progress bar and an ETA (change it
        later with ``run.set_total_steps``). ``wandb``: log this run to Weights & Biases (True or False); None
        follows the project's setting (``ProjectInfo.wandb``). Endor logs from its servers through the organization's
        W&B connection; ``run.info().wandb_url`` links to the W&B run.
        """
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
            "include_optimizer": include_optimizer,
            "lora": lora,
            "name": name,
            "tags": tags or [],
            "config": dict(config or {}),
            "user_metadata": user_metadata or {},
        }
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
        r = self._t.request(
            "POST",
            f"/v1/projects/{self._project}/runs",
            json=body,
            method_name="project.runs.create",
            idempotent=True,
        )
        run = Run(self._t, r, self._images)
        run._start_heartbeat()
        if wait:
            try:
                _wait_ready(run)
            except BaseException:
                logger.warning("run %s: stopped waiting for provisioning; closing it so its GPU is released", run.id)
                run._close_on_exit(failing=True)
                raise
        return run

    def get(self, run_id: str) -> Run:
        """A handle on an existing run, continuing its sequence numbers from the server. ``run.info_`` is the run as
        fetched, e.g. ``run.info_.step`` to resume from the right step; ``run.info()`` fetches it again."""
        return Run(self._t, self._t.request("GET", f"/v1/runs/{run_id}", method_name="project.runs.get"), self._images)

    def list(self, tag: str | None = None, limit: int | None = None, offset: int = 0) -> list[Run]:
        """The project's runs, newest first, optionally those carrying ``tag``: all of them, or at most ``limit``."""
        path = f"/v1/projects/{self._project}/runs"
        items = _list_all(self._t, path, {"tag": tag}, "project.runs.list", limit, offset)
        return [Run(self._t, x, self._images) for x in items]


class Models:
    """``project.models``: the project's base models (by id, e.g. ``"decider-2b"``) and the models saved by its runs.
    ``model`` arguments take a name (or a base id) or ``"<project>/<name>"``."""

    def __init__(self, transport: Transport, project: str) -> None:
        self._t, self._project, self._base = transport, project, f"/v1/projects/{project}/models"

    @staticmethod
    def _name(model: str) -> str:
        return model.split("/", 1)[1] if "/" in model else model

    def list(self, run_id: str | None = None) -> list[ModelInfo]:
        """The project's models: its base models first, in the order they were added (``kind == "base"``, ``name``
        the base id), then every saved, unexpired model, newest first (``kind == "saved"``); with ``run_id``, only
        the models that run saved."""
        items = _list_all(self._t, self._base, {"run_id": run_id}, "project.models.list")
        return [ModelInfo.model_validate(m) for m in items]

    def get(self, model: str) -> ModelInfo:
        """One model (a name or ``"<project>/<name>"``); ``NotFoundError`` if missing."""
        return ModelInfo.model_validate(
            self._t.request("GET", f"{self._base}/{self._name(model)}", method_name="project.models.get")
        )

    def set_ttl(self, model: str, ttl_seconds: int | None) -> ModelInfo:
        """Delete the model after ``ttl_seconds`` from now (1 hour to 10 years), or keep it with None."""
        r = self._t.request(
            "PATCH",
            f"{self._base}/{self._name(model)}",
            json={"ttl_seconds": ttl_seconds},
            method_name="project.models.set_ttl",
        )
        return ModelInfo.model_validate(r)

    def download(
        self, model: str, path: str | os.PathLike[str], include_optimizer: bool = False
    ) -> builtins.list[DownloadedFile]:
        """Write the model's files into the folder ``path`` (created if needed), to run the LoRA adapter (PEFT)
        yourself on the same base model: ``adapter_model.safetensors``, ``adapter_config.json``,
        ``endor_manifest.json``, ``readout.safetensors`` if the model has one, and ``optimizer.pt`` (Adam state, only
        to resume training) with ``include_optimizer=True``. Each file is fetched straight from storage over a
        short-lived link and checked against its size and SHA-256; a file that fails raises ``DownloadError`` and is
        not left behind. The links are never returned or logged."""
        from ._download import fetch_all

        r = self._t.request(
            "GET",
            f"{self._base}/{self._name(model)}/download",
            params={"include_optimizer": "true" if include_optimizer else "false"},
            method_name="project.models.download",
        )
        return fetch_all(r["files"], path, transport=self._t._transport, timeout=max(self._t.timeout, 60.0))

    def delete(self, model: str) -> None:
        """Delete a saved model's files and record. Permanent. A base model raises ``UnprocessableEntityError``:
        remove it with ``project.remove_base_model``."""
        self._t.request("DELETE", f"{self._base}/{self._name(model)}", method_name="project.models.delete")


# ---------------------------------------------------------------- helpers


def _base_list(base_models: Sequence[str] | None, removed: dict[str, Any], method: str) -> list[str]:
    """``base_models`` as a list; a clear TypeError for ``base_model=`` (replaced by ``base_models``) or a string."""
    if "base_model" in removed:
        b = removed.pop("base_model")
        hint = f"base_models=[{b!r}]" if isinstance(b, str) else "base_models=[...]"
        raise TypeError(f"{method}: base_model was replaced by base_models (a project can have several): pass {hint}")
    if removed:
        raise TypeError(f"{method}() got an unexpected keyword argument {next(iter(removed))!r}")
    if isinstance(base_models, str):
        raise TypeError(f"{method}: base_models is a list of base model ids: pass base_models=[{base_models!r}]")
    return list(base_models or [])


def _list_all(
    t: Transport,
    path: str,
    params: dict[str, Any],
    method_name: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[Any]:
    """Every item of a paged list (``{"items", "total"}``), or at most ``limit`` of them, starting at ``offset``."""
    out: list[Any] = []
    while limit is None or len(out) < limit:
        size = C.LIST_PAGE_SIZE if limit is None else min(C.LIST_PAGE_SIZE, limit - len(out))
        r = t.request("GET", path, params={**params, "limit": size, "offset": offset}, method_name=method_name)
        items = r["items"]
        out += items
        offset += len(items)
        if not items or offset >= r.get("total", offset):
            break
    return out


def _wait_ready(run: Run, log_every: float = 30.0) -> None:
    started = time.monotonic()
    if run.info_.status in ("ready", "idle", "closed", "failed"):
        return
    logger.info("run %s: provisioning a trainer for %s ...", run.id, run.info_.base_model)
    while True:
        try:
            run.ready.result(timeout=log_every)
            break
        except TimeoutError:
            logger.info("run %s: still provisioning (%.0fs)", run.id, time.monotonic() - started)
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
