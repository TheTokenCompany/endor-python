"""Continuous learning projects: Endor trains new versions of a project's model from its decisions and your
corrections, starting from the base model you choose when you create it.

    project = client.projects.get_or_create("tickets", base_model="decider-2b")
    client.system_one(state, questions, model=project.ref)            # "tickets": the production version
    client.system_one(state, questions, model=project.preview_ref)    # "tickets:preview": a new version
    project.versions()               # v1, v2, ... newest first: which one is in production, which is the preview
    project.promote("v3")            # put v3 in production (an older one: a rollback)
    project.set_auto_promote(True)   # every new version goes to production as soon as it is ready (off by default)

A project has one production version, which ``"<project>"`` answers with (its base model until you promote one). New
versions land in the preview, the newest version while it isn't in production, which ``"<project>:preview"`` answers
with, until you promote one; without a preview, that name raises ``NoPreviewError``. With auto-promote on, each new
version goes to production at once, so there is no preview. Custom models you train yourself are
``client.custom_models``.
"""

from __future__ import annotations

import builtins
import os
from typing import Any

from . import _constants as C
from ._http import Transport
from ._pages import list_all
from ._parse import parse
from .errors import NotFoundError
from .types import DownloadedFile, Learning, ProjectInfo, VersionInfo

__all__ = ["Projects", "Project"]


class Projects:
    """``client.projects``: create, fetch and list continuous learning projects."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def create(
        self, name: str, base_model: str | None = None, description: str | None = None, **removed: Any
    ) -> Project:
        """A new continuous learning project on ``base_model`` (an id from ``client.base_models()``, for example
        ``"decider-2b"``), chosen here for good. ``"<project>"`` answers with the base model until you promote a
        version. Auto-promote starts off: new versions wait in the preview (``set_auto_promote(True)`` changes that).

        Names are lowercase, ``[a-z0-9._-]``, up to 63 characters, and unique in the organization together with your
        custom models. Raises ``ConflictError`` if a project or custom model already has the name, and
        ``LimitReachedError`` when the org has as many projects as it may (``max_projects_per_org``)."""
        base_model = _base(base_model, removed, "projects.create")
        body: dict[str, Any] = {"name": name, "description": description, "base_model": base_model}
        r = self._t.request("POST", "/v1/projects", json=body, method_name="projects.create", idempotent=True)
        return Project(self._t, r, endpoint="POST /v1/projects")

    def get(self, name: str) -> Project:
        """Your project called ``name``; ``NotFoundError`` if there is none."""
        path = f"/v1/projects/{name}"
        return Project(self._t, self._t.request("GET", path, method_name="projects.get"), endpoint=f"GET {path}")

    def get_or_create(
        self, name: str, base_model: str | None = None, description: str | None = None, **removed: Any
    ) -> Project:
        """The project called ``name``, created on ``base_model`` if missing. Safe to call at the top of every script.
        Raises ``ValueError`` for an existing project on another base model (a project's base model is chosen once, so
        pick another name)."""
        base_model = _base(base_model, removed, "projects.get_or_create")
        try:
            p = self.get(name)
        except NotFoundError:
            return self.create(name, base_model, description)
        if p.info_.base_model != base_model:
            raise ValueError(
                f"project {name!r} learns on {p.info_.base_model}, not {base_model}; a project's base model is chosen "
                "once, so pick another name"
            )
        return p

    def list(self, limit: int | None = None, offset: int = 0) -> list[Project]:
        """Your projects, newest first: all of them, or at most ``limit`` starting at ``offset``."""
        items = list_all(self._t, "/v1/projects", {}, "projects.list", limit, offset)
        return [Project(self._t, p, endpoint="GET /v1/projects") for p in items]


class Project:
    """One continuous learning project. ``ref`` and ``preview_ref`` are what you pass as ``model``."""

    def __init__(self, transport: Transport, info: dict[str, Any], *, endpoint: str | None = None) -> None:
        self._t = transport
        self.info_ = parse(ProjectInfo, info, endpoint)
        """The project as last fetched (``project.info()`` refreshes it)."""
        self.name: str = self.info_.name

    def __repr__(self) -> str:
        return f"Project({self.name!r})"

    @property
    def ref(self) -> str:
        """``"<project>"``: the ``model`` that answers with the production version (the base model until you promote
        one)."""
        return self.name

    @property
    def preview_ref(self) -> str | None:
        """``"<project>:preview"``: the ``model`` that answers with the preview, the newest version while it isn't in
        production. None while the project has no preview (as last fetched: ``info()`` refreshes it), when that name
        raises ``NoPreviewError``."""
        return f"{self.name}:{C.PREVIEW}" if self.info_.preview else None

    @property
    def production(self) -> str | None:
        """The version in production (``"v4"``), which ``"<project>"`` answers with; None while its base model
        answers. As last fetched (``info()`` refreshes it)."""
        return self.info_.production

    @property
    def preview(self) -> str | None:
        """The preview version, the newest while it isn't in production; None without one. As last fetched."""
        return self.info_.preview

    @property
    def auto_promote(self) -> bool:
        """Whether each new version goes to production as soon as it is ready (off by default). As last fetched;
        ``set_auto_promote`` changes it."""
        return self.info_.auto_promote

    def info(self) -> ProjectInfo:
        """Fetch the project again: its production and preview versions, auto-promote and description."""
        path = f"/v1/projects/{self.name}"
        self.info_ = parse(ProjectInfo, self._t.request("GET", path, method_name="project.info"), f"GET {path}")
        return self.info_

    def learning(self) -> Learning:
        """The project's learning, which is always on: its status, the way to the next version (``progress``, 0 to
        1), what ``"<project>"`` answers with now, the latest training runs (each kept or not, with its accuracy and
        loss) and whether recent traffic changed (``drift``)."""
        path = f"/v1/projects/{self.name}/learning"
        return parse(Learning, self._t.request("GET", path, method_name="project.learning"), f"GET {path}")

    def versions(self) -> builtins.list[VersionInfo]:
        """Every version, newest first, with which one is in production, which one is the preview (the newest, while
        it isn't in production) and how each did when it was trained."""
        path = f"/v1/projects/{self.name}/versions"
        return [parse(VersionInfo, v, f"GET {path}") for v in list_all(self._t, path, {}, "project.versions")]

    def promote(self, version: str) -> ProjectInfo:
        """Put ``version`` (``"v3"``) in production: ``"<project>"`` answers with it from now on. Promoting a version
        older than the newest is a rollback: it turns auto-promote off, so the newest becomes the preview and new
        versions only reach ``"<project>:preview"`` until you promote again or call ``set_auto_promote(True)``.
        Promoting the production version changes nothing. ``NotFoundError`` for a version the project doesn't have."""
        path = f"/v1/projects/{self.name}/versions/{version}/promote"
        self.info_ = parse(ProjectInfo, self._t.request("POST", path, method_name="project.promote"), f"POST {path}")
        return self.info_

    def set_auto_promote(self, on: bool) -> ProjectInfo:
        """Turn auto-promote on (the newest version goes to production at once, and every new version after it, so
        there is no preview) or off, the default (the production version stays until you promote one; newer versions
        are the preview)."""
        path = f"/v1/projects/{self.name}"
        r = self._t.request("PATCH", path, json={"auto_promote": bool(on)}, method_name="project.set_auto_promote")
        self.info_ = parse(ProjectInfo, r, f"PATCH {path}")
        return self.info_

    def update(self, *, description: str | None = None) -> ProjectInfo:
        """Change the project's description. Its base model never changes; auto-promote changes with
        ``set_auto_promote``."""
        if description is None:
            raise ValueError("nothing to update: pass description")
        path = f"/v1/projects/{self.name}"
        r = self._t.request("PATCH", path, json={"description": description}, method_name="project.update")
        self.info_ = parse(ProjectInfo, r, f"PATCH {path}")
        return self.info_

    def download(
        self, version: str, path: str | os.PathLike[str], include_optimizer: bool = False
    ) -> builtins.list[DownloadedFile]:
        """Write a version's files into the folder ``path`` (created if needed; ``NotFoundError`` for a version without
        stored weights), to run the LoRA adapter (PEFT)
        yourself on the project's base model: ``adapter_model.safetensors``, ``adapter_config.json``,
        ``endor_manifest.json`` and ``readout.safetensors`` if it has one. Each file is fetched straight from storage
        over a short-lived link and checked against its size and SHA-256 (``DownloadError`` otherwise; nothing is left
        behind). ``include_optimizer=True`` adds ``optimizer.pt``; a version saved without optimizer state (Endor's
        training doesn't store it) raises ``DownloadError`` before anything is written."""
        from ._download import fetch_all

        r = self._t.request(
            "GET",
            f"/v1/projects/{self.name}/versions/{version}/download",
            params={"include_optimizer": "true" if include_optimizer else "false"},
            method_name="project.download",
        )
        return fetch_all(
            r["files"],
            path,
            what=f"{self.name} {version}",
            include_optimizer=include_optimizer,
            transport=self._t._transport,
            timeout=max(self._t.timeout, 60.0),
        )

    def delete(self) -> None:
        """Delete the project and its versions. Permanent: ``"<project>"`` stops answering, and its name is free at
        once."""
        self._t.request("DELETE", f"/v1/projects/{self.name}", method_name="project.delete")


def _base(base_model: str | None, removed: dict[str, Any], method: str) -> str:
    """``base_model``, with a clear TypeError for the release before's ``kind`` and ``base_models``."""
    if "base_models" in removed:
        b = removed.pop("base_models")
        one = b[0] if isinstance(b, (list, tuple)) and len(b) == 1 else None
        hint = f"base_model={one!r}" if isinstance(one, str) else "base_model=..."
        raise TypeError(f"{method}: a project has one base model now: pass {hint}")
    if "kind" in removed:
        removed.pop("kind")
        raise TypeError(
            f"{method}: kind was removed: every project is a continuous learning project. Train your own models with "
            "client.custom_models.create(name, base_model=...)"
        )
    if removed:
        raise TypeError(f"{method}() got an unexpected keyword argument {next(iter(removed))!r}")
    if not isinstance(base_model, str) or not base_model:
        raise TypeError(f'{method}: pass the project\'s base model, e.g. base_model="decider-2b"')
    return base_model
