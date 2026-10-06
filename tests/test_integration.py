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
    assert live.whoami().user_id
    assert base_model in {m.name for m in live.models.list().models}


def test_decision(live: endor.EndorClient, base_model: str) -> None:
    res = live.system_one({"body": "I was charged twice"}, {"dept": DEPT, "urgent": URGENT}, model=base_model)
    assert res.model == base_model
    assert abs(sum(res.choices["dept"].probabilities.values()) - 1) < 1e-3
    assert 0 <= res.nouls["urgent"].noul <= 1


def test_tiny_training_loop(live: endor.EndorClient, base_model: str) -> None:
    project = live.projects.create(f"sdk-it-{uuid.uuid4().hex[:8]}")
    try:
        rows = [
            {
                "state": {"body": f"ticket {i}"},
                "questions": {"dept": DEPT, "urgent": URGENT},
                "labels": {"dept": "billing" if i % 2 else "technical", "urgent": bool(i % 3)},
            }
            for i in range(8)
        ]
        project.datasets.upload("train", rows)
        with project.runs.create(base_model, rank=4) as run:
            datums = endor.data.rows_to_datums(rows)
            before = run.forward(datums).result()
            fb = run.forward_backward(datums)
            opt = run.optim_step(learning_rate=1e-4)
            out, step = endor.gather(fb, opt)
            assert out.metrics["n"] == len(datums) and step.step == 1
            after = run.forward(datums).result()
            assert len(before.probabilities) == len(after.probabilities) == len(datums)
            model = run.save_checkpoint("v1").result()
        assert model == f"{project.name}/v1"
        res = live.system_one(rows[0]["state"], {"dept": DEPT}, model=model)
        assert res.model == model
        assert any(m.name == "v1" for m in project.models.list())
        ev = project.evaluate(model, "train").result()
        assert ev.status == "completed"
        project.models.delete("v1")
    finally:
        project.delete()
