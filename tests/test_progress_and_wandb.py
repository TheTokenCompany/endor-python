"""Training progress (total_steps, set_total_steps, RunInfo.progress and eta_seconds, the recipe's progress line) and
the Weights & Biases flags: the project setting, the per-run override, and the run's W&B link."""

from __future__ import annotations

from typing import Any

import pytest

from endor import ConflictError, EndorClient, RunInfo
from endor.recipes.supervised import SupervisedConfig, train

from .conftest import requests_to, rows, unique
from .fake_api import FakeEndor


def test_total_steps_progress_and_eta(client: EndorClient, fake: FakeEndor) -> None:
    project = client.projects.create(unique("prog"), base_models=["jev-9b"])
    with project.runs.create(total_steps=10) as run:
        assert requests_to(fake, "/runs", "POST")[-1].body["total_steps"] == 10
        assert run.info_.total_steps == 10 and run.info_.progress == 0.0 and run.info_.ready_at is not None
        info = run.set_total_steps(40)
        assert fake.requests[-1].method == "PATCH" and fake.requests[-1].body == {"total_steps": 40}
        assert info.total_steps == 40 and run.info().total_steps == 40
        assert run.set_total_steps(None).total_steps is None and run.info_.progress is None
        with pytest.raises(ValueError):
            run.set_total_steps(0)
    with pytest.raises(ValueError):
        project.runs.create(total_steps=0)
    i = RunInfo.model_validate(
        {
            "id": "r",
            "project": "p",
            "base_model": "b",
            "status": "ready",
            "created_at": "2026-10-07T00:00:00Z",
            "step": 30,
            "total_steps": 120,
            "seconds_per_step": 2.0,
        }
    )
    assert i.progress == 0.25 and i.eta_seconds == 180.0
    assert i.model_copy(update={"seconds_per_step": None}).eta_seconds is None


def test_recipe_sets_total_steps_and_prints_progress(client: EndorClient, fake: FakeEndor, capsys: Any) -> None:
    cfg = SupervisedConfig(project=unique("sft"), base_model="jev-9b", batch_size=8, epochs=2, eval_base=False)
    result = train(cfg, rows(40), client=client)
    body = requests_to(fake, "/runs", "POST")[-1].body
    total = 2 * 9  # 72 train datums (36 rows x 2 questions) after a 10% holdout, in batches of 8, twice
    assert body["total_steps"] == total
    err = capsys.readouterr().err
    assert f"run {result.run_id}: step {total}/{total} (100%) · loss" in err
    assert "left" in err  # the earlier lines carry an ETA


def test_wandb_flags(client: EndorClient, fake: FakeEndor) -> None:
    project = client.projects.create(unique("wb"), base_models=["jev-9b"])
    assert project.info_.wandb is not None and project.info_.wandb.enabled is False
    with pytest.raises(ConflictError):  # the org hasn't connected W&B
        project.update(wandb={"enabled": True})
    fake.wandb_connected = True
    info = project.update(wandb={"enabled": True, "project": "tix"})
    assert info.wandb is not None and (info.wandb.enabled, info.wandb.project) == (True, "tix")
    with project.runs.create() as run:
        assert "wandb" not in requests_to(fake, "/runs", "POST")[-1].body  # the project decides
        assert run.info_.wandb is True and run.info_.wandb_url is None
    with project.runs.create(wandb=False) as run:
        assert requests_to(fake, "/runs", "POST")[-1].body["wandb"] is False
        assert run.info_.wandb is False
