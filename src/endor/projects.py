"""Projects: the unit of work. A project, keyed by its name, holds datasets, runs, models and evaluations.

project = client.projects.get_or_create("tickets", base_model="decider-2b")   # kind="custom"
project.datasets.upload("train", rows)
run = project.runs.create(base_model="decider-2b")
project.models.list()                      # "tickets/base", then every saved model: "tickets/<name>"
project.set_live("v1")                     # model="tickets" now answers with "tickets/v1"
project.evaluate("base", "heldout").result()   # names resolve in the project: "base" is "tickets/base"
"""

from __future__ import annotations

import builtins
import os
import subprocess
import time
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from . import _constants as C
from . import data as D
from ._http import Transport
from ._log import logger
from .errors import NotFoundError
from .futures import APIFuture
from .runs import Run
from .types import (
    DatasetInfo,
    DecisionRow,
    DownloadedFile,
    Evaluation,
    LoraConfig,
    ModelInfo,
    ProjectInfo,
)

ProjectKind = Literal["custom", "managed"]
"""``custom``: you train models with the SDK. ``managed``: Endor trains new versions from the project's decisions."""


__all__ = ["Projects", "Project", "Datasets", "Runs", "Models", "MAX_UPLOAD_ROWS"]

MAX_UPLOAD_ROWS = C.MAX_UPLOAD_ROWS


class Projects:
    """``client.projects``: create, fetch and list projects."""

    def __init__(self, transport: Transport, capture: bool) -> None:
        self._t, self._capture = transport, capture

    def create(
        self,
        name: str,
        description: str | None = None,
        *,
        base_model: str | None = None,
        kind: ProjectKind = "custom",
    ) -> Project:
        """A new project. Names are lowercase, ``[a-z0-9._-]``, up to 63 characters, and permanent (they are part
        of every model id). Raises ``ConflictError`` if you already have one with that name, and
        ``LimitReachedError`` (a ``ConflictError``) when the org has as many projects as it may.

        ``kind`` is ``"custom"`` (you train models with the SDK) or ``"managed"`` (Endor trains new versions from the
        project's decisions; its decisions cost 50% more). ``base_model`` (for example ``"decider-2b"``) is the
        project's base model, and ``model="<name>"`` answers with it at once. A managed project needs it; a custom
        project may leave it out, and its first ``runs.create(base_model=...)`` sets it. Neither can change later."""
        if kind == "managed" and base_model is None:
            raise ValueError("a managed project needs a base_model")
        body: dict[str, Any] = {"name": name, "description": description, "kind": kind, "base_model": base_model}
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
        base_model: str | None = None,
        kind: ProjectKind = "custom",
    ) -> Project:
        """The project called ``name``, created if missing (with these settings). Safe to call at the top of every
        script; an existing project's description is left as it is. Raises ``ValueError`` if the existing project
        has another ``kind``, or another ``base_model`` once both are set: neither can change, so use another project
        name."""
        try:
            p = self.get(name)
        except NotFoundError:
            return self.create(name, description, base_model=base_model, kind=kind)
        other_base = base_model is not None and p.info_.base_model is not None and p.info_.base_model != base_model
        if p.info_.kind != kind or other_base:
            raise ValueError(
                f"project {name!r} exists as a {p.info_.kind} project on {p.info_.base_model}, not a {kind} project "
                f"on {base_model}; a project's kind and base model can't change, so pick another name"
            )
        return p

    def list(self, limit: int | None = None, offset: int = 0) -> list[Project]:
        """Your projects, newest first: all of them, or at most ``limit`` starting at ``offset``."""
        items = _list_all(self._t, "/v1/projects", {}, "projects.list", limit, offset)
        return [Project(self._t, p, self._capture) for p in items]


class Project:
    """One project. ``.datasets``, ``.runs`` and ``.models`` hold what it contains."""

    def __init__(self, transport: Transport, info: dict[str, Any], capture: bool = True) -> None:
        self._t = transport
        self.info_ = ProjectInfo.model_validate(info)
        """Name, description and counts as last fetched."""
        self.name: str = self.info_.name
        self.datasets = Datasets(transport, self.name)
        self.runs = Runs(transport, self.name, capture)
        self.models = Models(transport, self.name)

    def __repr__(self) -> str:
        return f"Project({self.name!r})"

    def info(self) -> ProjectInfo:
        """Refresh the project's counts."""
        self.info_ = ProjectInfo.model_validate(
            self._t.request("GET", f"/v1/projects/{self.name}", method_name="project.info")
        )
        return self.info_

    def set_live(self, model: str) -> ProjectInfo:
        """Make ``model`` the live model: what ``model="<project>"`` answers with. ``model`` is a saved model's name,
        ``"<project>/<name>"``, or ``"base"`` for the project's base model. Custom projects only: a managed project
        always serves its newest version (``WrongProjectKindError``). Raises ``NotFoundError`` for an unknown model."""
        self.info_ = ProjectInfo.model_validate(
            self._t.request(
                "POST", f"/v1/projects/{self.name}/live", json={"model": model}, method_name="project.set_live"
            )
        )
        return self.info_

    def update(
        self,
        *,
        description: str | None = None,
        paused: bool | None = None,
    ) -> ProjectInfo:
        """Change the project's settings; arguments left as None are unchanged. The kind and base model never change.

        - ``description``: free text.
        - ``paused``: managed projects only (``WrongProjectKindError`` for a custom one). A paused project stops
          learning and keeps serving its newest version; its decisions keep the managed price."""
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if paused is not None:
            body["paused"] = paused
        if not body:
            raise ValueError("nothing to update: pass description or paused")
        self.info_ = ProjectInfo.model_validate(
            self._t.request("PATCH", f"/v1/projects/{self.name}", json=body, method_name="project.update")
        )
        return self.info_

    def evaluate(self, model: str, dataset: str, run_id: str | None = None) -> APIFuture[Evaluation]:
        """Score a model on one of this project's datasets, server-side. ``model`` is ``"base"`` (this project's
        base model), a model name of this project, this project's name (its live model) or a full model id
        (``"<project>"``, ``"<project>/base"``, ``"<project>/<name>"``). A bare base model id is refused. The
        evaluation records the full id of the model that answered: ``"base"`` becomes ``"<project>/base"``.

        The result's ``results`` has ``n``, then ``overall``, ``by_type`` and ``by_question``, each with accuracy,
        NLL, Brier, ECE, mean confidence and ``selective`` (``{"0.5": {"accuracy", "coverage"}, ...}`` for the
        thresholds 0.5, 0.7, 0.9 and 0.95), and ``rows_url``, a link to every scored question that expires after 1
        hour. With ``run_id``, the evaluation appears on that run's page.
        """
        body = {"model": model, "dataset": dataset, "training_run_id": run_id}
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

    def __init__(self, transport: Transport, project: str) -> None:
        self._t, self._base = transport, f"/v1/projects/{project}/datasets"

    def upload(self, name: str, rows: Iterable[Any]) -> DatasetInfo:
        """Upload decision rows (README, "Data format"). Rows may be dicts or ``DecisionRow`` objects; the other
        supported row formats are converted. At most 50,000 rows per dataset. Datasets are immutable: to change
        one, upload under a new name."""
        wire = [D.to_row(r).to_wire() for r in rows]
        if not wire:
            raise ValueError("no rows to upload")
        if len(wire) > MAX_UPLOAD_ROWS:
            raise ValueError(f"at most {MAX_UPLOAD_ROWS} rows per dataset (got {len(wire)}); split the data")
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

    def __init__(self, transport: Transport, project: str, capture: bool) -> None:
        self._t, self._project, self._capture = transport, project, capture

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
        wait: bool = True,
    ) -> Run:
        """Start a run: a fresh adapter on the project's base model, or one warm-started from ``from_model`` (a model
        of this project, as ``"name"`` or ``"<project>/name"``; with ``include_optimizer=True`` training resumes
        exactly).

        LoRA settings left as None default to rank 16, alpha 32, attention and MLP, no readout for a fresh run. With
        ``from_model`` they are inherited from the saved model, and only the ones you pass are sent (a value that
        conflicts with the model is a 422).

        Provisioning a trainer usually takes under a minute, and the GPU is billed from the moment it is requested. With
        ``wait=True`` this blocks until the run is ready and logs progress on the ``endor`` logger; if the wait is
        interrupted (Ctrl-C, an error) the run is closed so its GPU is released. With ``wait=False`` it returns at
        once and ``run.ready`` is the future.

        Close every run you create (``with project.runs.create(...) as run:``): a run holds its GPU until closed or
        idle for 15 minutes, and an org can have at most 4 runs that aren't closed, idle ones included (a fifth
        raises ``LimitReachedError``). ``base_model`` may be left out: a run always trains on the project's base model,
        and another base is a 422 (``UnprocessableEntityError``). Custom projects only: a managed project raises
        ``WrongProjectKindError`` (Endor trains it).
        ``config`` is free-form and shown on the dashboard; when the client was created with ``capture=True`` the
        SDK adds the LoRA settings and the current git commit (never file contents).
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
        run = Run(self._t, r)
        if wait:
            try:
                _wait_ready(run)
            except BaseException:
                logger.warning("run %s: stopped waiting for provisioning; closing it so its GPU is released", run.id)
                run._close_on_exit(failing=True)
                raise
        return run

    def get(self, run_id: str) -> Run:
        """A handle on an existing run, continuing its sequence numbers from the server."""
        return Run(self._t, self._t.request("GET", f"/v1/runs/{run_id}", method_name="project.runs.get"))

    def list(self, tag: str | None = None, limit: int | None = None, offset: int = 0) -> list[Run]:
        """The project's runs, newest first, optionally those carrying ``tag``: all of them, or at most ``limit``."""
        path = f"/v1/projects/{self._project}/runs"
        items = _list_all(self._t, path, {"tag": tag}, "project.runs.list", limit, offset)
        return [Run(self._t, x) for x in items]


class Models:
    """``project.models``: the project's base model (``"base"``) and the models saved by its runs. ``model``
    arguments take a name or ``"<project>/<name>"``."""

    def __init__(self, transport: Transport, project: str) -> None:
        self._t, self._project, self._base = transport, project, f"/v1/projects/{project}/models"

    @staticmethod
    def _name(model: str) -> str:
        return model.split("/", 1)[1] if "/" in model else model

    def list(self, run_id: str | None = None) -> list[ModelInfo]:
        """The project's models: its base model first (``name == "base"``), then every saved, unexpired model, newest
        first; optionally only those saved by ``run_id``."""
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
        """Delete the model's files and record. Permanent. The base model can't be deleted."""
        self._t.request("DELETE", f"{self._base}/{self._name(model)}", method_name="project.models.delete")


# ---------------------------------------------------------------- helpers


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
