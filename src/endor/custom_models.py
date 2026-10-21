"""Custom models: models you train yourself with the SDK, called by their name while live.

    run = client.custom_models.create("router-2", base_model="decider-2b")   # a run that saves as "router-2"
    with run:
        ...                                          # forward_backward, optim_step (see ``endor.Run``)
        run.save_checkpoint().result()               # "router-2" is saved, inactive
    client.custom_models.promote("router-2")         # live: model="router-2" answers
    client.custom_models.list()                      # live ones first, then the inactive ones

Each save is a new model with its own name, unique in the organization together with the continuous learning
projects. A saved model starts inactive: decisions refuse it (``ModelNotLiveError``), but it can be downloaded and
evaluated. An organization has a few live models at once (``max_live_models_per_org`` in ``client.whoami().limits``);
when every slot is taken, ``promote(name, replace=...)`` swaps one out in a single step.
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
    """``client.custom_models``: list, train, promote and download your custom models."""

    def __init__(self, transport: Transport, runs: Runs) -> None:
        self._t, self._runs, self._base = transport, runs, "/v1/custom_models"

    def list(
        self, live: bool | None = None, limit: int | None = None, offset: int = 0
    ) -> builtins.list[CustomModelInfo]:
        """Your custom models: the live ones first (newest promoted first), then the inactive ones, newest first.
        ``live=True`` or ``live=False`` keeps only one of the two."""
        params = {"live": None if live is None else ("true" if live else "false")}
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
        ``from_model``, ``total_steps``, ``wait``, ...). Close the run when you are done
        (``with client.custom_models.create(...) as run:``)."""
        if "model" in run_options:
            raise TypeError("custom_models.create: the model's name is the first argument")
        return self._runs.create(base_model, model=name, name=run_name, **run_options)

    def promote(self, name: str, replace: str | None = None) -> CustomModelInfo:
        """Make the model live: decisions may call it by name (``model=name``). A live model stays live (nothing
        changes). When every live slot is taken, ``LiveModelsFullError`` names the live models, unless ``replace``
        names one: it is demoted in the same step. ``NotFoundError`` for an unknown model, or a ``replace`` that isn't
        a live custom model."""
        body = {"replace": replace} if replace is not None else {}
        return CustomModelInfo.model_validate(
            self._t.request("POST", f"{self._base}/{name}/promote", json=body, method_name="custom_models.promote")
        )

    def demote(self, name: str) -> CustomModelInfo:
        """Make the model inactive: decisions refuse it (``ModelNotLiveError``) and its live slot is free. An
        inactive model stays inactive (nothing changes)."""
        return CustomModelInfo.model_validate(
            self._t.request("POST", f"{self._base}/{name}/demote", method_name="custom_models.demote")
        )

    def set_ttl(self, name: str, ttl_seconds: int | None) -> CustomModelInfo:
        """Delete an inactive model after ``ttl_seconds`` from now (1 hour to 10 years), or keep it with None. Live
        models never expire (``ModelIsLiveError``)."""
        return CustomModelInfo.model_validate(
            self._t.request(
                "PATCH", f"{self._base}/{name}", json={"ttl_seconds": ttl_seconds}, method_name="custom_models.set_ttl"
            )
        )

    def download(
        self, name: str, path: str | os.PathLike[str], include_optimizer: bool = False
    ) -> builtins.list[DownloadedFile]:
        """Write the model's files into the folder ``path`` (created if needed; ``NotFoundError`` for a model without
        stored weights), live or not, to run the LoRA adapter
        (PEFT) yourself on the same base model: ``adapter_model.safetensors``, ``adapter_config.json``,
        ``endor_manifest.json``, ``readout.safetensors`` if the model has one, and ``optimizer.pt`` (Adam state, only
        to resume training) with ``include_optimizer=True``. Each file is fetched straight from storage over a
        short-lived link and checked against its size and SHA-256; a file that fails raises ``DownloadError`` and is
        not left behind. The links are never returned or logged."""
        from ._download import fetch_all

        r = self._t.request(
            "GET",
            f"{self._base}/{name}/download",
            params={"include_optimizer": "true" if include_optimizer else "false"},
            method_name="custom_models.download",
        )
        return fetch_all(r["files"], path, transport=self._t._transport, timeout=max(self._t.timeout, 60.0))

    def delete(self, name: str) -> None:
        """Delete an inactive model's files and record; its name is free at once. Permanent. A live model raises
        ``ModelIsLiveError`` (demote it first)."""
        self._t.request("DELETE", f"{self._base}/{name}", method_name="custom_models.delete")
