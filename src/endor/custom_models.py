"""Custom models: models you train yourself with the SDK, called by their name while in production.

    run = client.custom_models.create("router-2", base_model="decider-2b")   # a run that saves as "router-2"
    with run:
        ...                                          # forward_backward, optim_step (see ``endor.Run``)
        run.save_checkpoint().result()               # "router-2" is saved, in the archive
    client.custom_models.promote("router-2")         # production: model="router-2" answers
    client.custom_models.list()                      # production ones first, then the archived ones
    client.custom_models.archive("router-2")         # back to the archive: decisions refuse it

Each save is a new model with its own name, unique in the organization together with the continuous learning
projects. A saved model starts in the archive: decisions refuse it (``ModelNotInProductionError``), but it can be
downloaded, evaluated, trained from and promoted. An organization has a few models in production at once
(``max_production_models_per_org`` in ``client.whoami().limits``); when every slot is taken,
``promote(name, replace=...)`` swaps one out in a single step.
"""

from __future__ import annotations

import builtins
import os
from typing import Any

from ._http import Transport
from ._pages import list_all
from .runs import Run
from .training import Runs
from .types import CustomModelInfo, DownloadedFile

__all__ = ["CustomModels"]


class CustomModels:
    """``client.custom_models``: list, train, promote, archive and download your custom models."""

    def __init__(self, transport: Transport, runs: Runs) -> None:
        self._t, self._runs, self._base = transport, runs, "/v1/custom_models"

    def list(
        self, production: bool | None = None, limit: int | None = None, offset: int = 0
    ) -> builtins.list[CustomModelInfo]:
        """Your custom models: the ones in production first (in the order they were promoted), then the archived ones,
        newest first. ``production=True`` or ``production=False`` keeps only one of the two."""
        params = {"production": None if production is None else ("true" if production else "false")}
        items = list_all(self._t, self._base, params, "custom_models.list", limit, offset)
        return [CustomModelInfo.model_validate(m) for m in items]

    def get(self, name: str) -> CustomModelInfo:
        """One custom model; ``NotFoundError`` if missing."""
        return CustomModelInfo.model_validate(
            self._t.request("GET", f"{self._base}/{name}", method_name="custom_models.get")
        )

    def create(
        self, name: str, base_model: str | None = None, *, run_name: str | None = None, **run_options: Any
    ) -> Run:
        """Start training a new custom model called ``name`` on ``base_model``: a run whose
        ``run.save_checkpoint()`` saves under ``name``. The name must be free (``ConflictError`` if a custom model or
        project has it). ``run_name`` labels the run itself. ``run_options`` are ``client.runs.create``'s (``rank``,
        ``from_model``, ``total_steps``, ``wait``, ``progress``, ...). Close the run when you are done
        (``with client.custom_models.create(...) as run:``)."""
        if "model" in run_options:
            raise TypeError("custom_models.create: the model's name is the first argument")
        return self._runs.create(base_model, model=name, name=run_name, **run_options)

    def promote(self, name: str, replace: str | None = None) -> CustomModelInfo:
        """Promote the model to production: decisions may call it by name (``model=name``) at once. It is
        ``serving="deploying"`` until a decision server has loaded it (the first decisions may wait for that), then
        ``"live"``. A model in production stays there (nothing changes), except one whose loading failed
        (``serving="failed"``): it is deployed again. When every production slot is taken, ``ProductionFullError``
        names the models in production, unless ``replace`` names one: it moves to the archive in the same step.
        ``NotFoundError`` for an unknown model, or a ``replace`` that isn't in production."""
        body = {"replace": replace} if replace is not None else {}
        return CustomModelInfo.model_validate(
            self._t.request("POST", f"{self._base}/{name}/promote", json=body, method_name="custom_models.promote")
        )

    def archive(self, name: str) -> CustomModelInfo:
        """Move the model to the archive: decisions refuse it (``ModelNotInProductionError``) and its production slot
        is free. It can still be downloaded, evaluated, trained from and promoted again. An archived model stays
        archived (nothing changes)."""
        return CustomModelInfo.model_validate(
            self._t.request("POST", f"{self._base}/{name}/demote", method_name="custom_models.archive")
        )

    def set_ttl(self, name: str, ttl_seconds: int | None) -> CustomModelInfo:
        """Delete an archived model after ``ttl_seconds`` from now (1 hour to 10 years), or keep it with None. Models
        in production never expire (``ModelInProductionError``)."""
        return CustomModelInfo.model_validate(
            self._t.request(
                "PATCH", f"{self._base}/{name}", json={"ttl_seconds": ttl_seconds}, method_name="custom_models.set_ttl"
            )
        )

    def download(
        self, name: str, path: str | os.PathLike[str], include_optimizer: bool = False
    ) -> builtins.list[DownloadedFile]:
        """Write the model's files into the folder ``path`` (created if needed; ``NotFoundError`` for a model without
        stored weights), in production or archived, to run the LoRA adapter (PEFT) yourself on the same base model:
        ``adapter_model.safetensors``, ``adapter_config.json``, ``endor_manifest.json``, ``readout.safetensors`` if the
        model has one, and ``optimizer.pt`` (Adam state, only to resume training) with ``include_optimizer=True``.
        Each file is fetched straight from storage over a short-lived link and checked against its size and SHA-256;
        a file that fails raises ``DownloadError`` and is not left behind. The links are never returned or logged."""
        from ._download import fetch_all

        r = self._t.request(
            "GET",
            f"{self._base}/{name}/download",
            params={"include_optimizer": "true" if include_optimizer else "false"},
            method_name="custom_models.download",
        )
        return fetch_all(r["files"], path, transport=self._t._transport, timeout=max(self._t.timeout, 60.0))

    def delete(self, name: str) -> None:
        """Delete an archived model's files and record; its name is free at once. Permanent. A model in production
        raises ``ModelInProductionError`` (archive it first)."""
        self._t.request("DELETE", f"{self._base}/{name}", method_name="custom_models.delete")
