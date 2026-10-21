"""The SDK against a real Endor deployment. Skipped unless ENDOR_TEST_BASE_URL and ENDOR_TEST_API_KEY are set.

ENDOR_TEST_BASE_URL=https://... ENDOR_TEST_API_KEY=edk_... pytest tests/test_integration.py
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

import endor
from endor import Choice, Noul

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ENDOR_TEST_BASE_URL") and os.environ.get("ENDOR_TEST_API_KEY")),
    reason="set ENDOR_TEST_BASE_URL and ENDOR_TEST_API_KEY to run against a live API",
)

DEPT = Choice(instructions="Route the ticket", criteria={"billing": "Payments and refunds", "technical": None})
URGENT = Noul(instructions="Does this need an answer today?")


@pytest.fixture(scope="module")
def live() -> Iterator[endor.EndorClient]:
    with endor.EndorClient(api_key=os.environ["ENDOR_TEST_API_KEY"], base_url=os.environ["ENDOR_TEST_BASE_URL"]) as c:
        yield c


@pytest.fixture(scope="module")
def base_model(live: endor.EndorClient) -> str:
    bases = [b for b in live.base_models() if b.trainable]
    assert bases, "no trainable base model in the catalog"
    return bases[-1].id


def test_whoami_and_catalog(live: endor.EndorClient, base_model: str) -> None:
    me = live.whoami()
    assert me.org_id and me.limits["max_active_runs"] >= 1
    assert base_model not in {m.name for m in live.models.list().models}  # bases are called through a project


def test_decision(live: endor.EndorClient, base_model: str) -> None:
    project = live.projects.get_or_create(f"sdk-it-{uuid.uuid4().hex[:8]}", base_model=base_model)
    try:
        assert project.info_.base_model == base_model and project.info_.production is None
        assert project.name in {p.name for p in live.projects.list()}
        res = live.system_one({"body": "I was charged twice"}, {"dept": DEPT, "urgent": URGENT}, model=project.ref)
        assert res.model == project.name
        assert abs(sum(res.choices["dept"].probabilities.values()) - 1) < 1e-3
        assert 0 <= res.nouls["urgent"].noul <= 1
        assert project.ref in {m.name for m in live.models.list().models} and project.preview_ref is None
        assert project.versions() == []
    finally:
        project.delete()


def test_tiny_training_loop(live: endor.EndorClient) -> None:
    name = f"sdk-it-{uuid.uuid4().hex[:8]}"
    rows = [
        {
            "state": {"body": f"ticket {i}"},
            "questions": {"dept": DEPT, "urgent": URGENT},
            "labels": {"dept": "billing" if i % 2 else "technical", "urgent": bool(i % 3)},
        }
        for i in range(8)
    ]
    live.datasets.upload(name, rows)
    try:
        with live.custom_models.create(name, base_model="jev-9b", rank=4) as run:
            datums = endor.data.rows_to_datums(rows)
            before = run.forward(datums).result()
            fb = run.forward_backward(datums)
            opt = run.optim_step(learning_rate=1e-4)
            out, step = endor.gather(fb, opt)
            assert out.metrics["n"] == len(datums) and step.step == 1
            after = run.forward(datums).result()
            assert len(before.probabilities) == len(after.probabilities) == len(datums)
            model = run.save_checkpoint().result()
        assert model == name and not live.custom_models.get(name).live
        with pytest.raises(endor.ModelNotLiveError):
            live.system_one(rows[0]["state"], {"dept": DEPT}, model=model)
        ev = live.evaluations.create(name, name).result()  # an inactive model can be evaluated
        assert ev.status == "completed" and ev.model == model
        live.custom_models.promote(name)
        try:
            assert live.system_one(rows[0]["state"], {"dept": DEPT}, model=model).model == model
            assert live.custom_models.get(name).loss is not None
        finally:
            live.custom_models.demote(name)
        live.custom_models.delete(name)
    finally:
        live.datasets.delete(name)
