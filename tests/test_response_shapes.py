"""A 2xx whose body doesn't have the expected shape raises ResponseValidationError (an EndorError) naming the endpoint
and the field, from every resource, never pydantic's ValidationError."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pydantic
import pytest

import endor
from endor import Datum, Noul, Target

from .conftest import URGENT
from .fake_api import FakeEndor


def bad(fake: FakeEndor, body: Any) -> None:
    """The next request is answered 200 with ``body``."""
    fake.fail_next.append(httpx.Response(200, json=body))


def raises_shape_error(call: Callable[[], Any], endpoint: str, field: str) -> endor.ResponseValidationError:
    with pytest.raises(endor.ResponseValidationError) as e:
        call()
    assert isinstance(e.value, endor.EndorError) and not isinstance(e.value, pydantic.ValidationError)
    assert e.value.endpoint == endpoint and e.value.status == 200
    assert f"invalid response data: {field}:" in e.value.message
    return e.value


def test_projects(client: endor.EndorClient, fake: FakeEndor) -> None:
    bad(fake, {"name": "tickets"})
    raises_shape_error(lambda: client.projects.get("tickets"), "GET /v1/projects/tickets", "base_model")
    p = client.projects.create("tickets", base_model="jev-9b")
    bad(fake, {"status": "learning"})
    raises_shape_error(p.learning, "GET /v1/projects/tickets/learning", "serving")


def test_custom_models(client: endor.EndorClient, fake: FakeEndor) -> None:
    bad(fake, {"items": [{"name": "router-2", "base_model": "jev-9b", "created_at": "never"}], "total": 1})
    raises_shape_error(client.custom_models.list, "GET /v1/custom_models", "created_at")


def test_datasets(client: endor.EndorClient, fake: FakeEndor) -> None:
    bad(fake, {"name": "train", "created_at": "2026-10-21T00:00:00Z"})
    raises_shape_error(lambda: client.datasets.get("train"), "GET /v1/datasets/train", "n_rows")


def test_evaluations(client: endor.EndorClient, fake: FakeEndor) -> None:
    bad(fake, {"id": "evl_1", "model": "jev-9b"})
    raises_shape_error(lambda: client.evaluations.get("evl_1"), "GET /v1/evaluations/evl_1", "source")


def test_runs(client: endor.EndorClient, fake: FakeEndor) -> None:
    bad(fake, {"id": "run_1", "status": "ready"})
    raises_shape_error(lambda: client.runs.get("run_1"), "GET /v1/runs/run_1", "base_model")


def test_corrections(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    res = client.system_one({"body": "charged twice"}, {"urgent": URGENT}, model=decider)
    bad(fake, {"request_id": res.id, "questions": {"urgent": {"correct": "maybe"}}})
    raises_shape_error(
        lambda: client.corrections.create(res, {"urgent": {"correct": True}}),
        "POST /v1/corrections",
        "questions.urgent.correct",
    )


def test_futures(client: endor.EndorClient, fake: FakeEndor) -> None:
    datum = Datum(state="x", question=Noul(instructions="?"), target=Target(label="true"))
    with client.runs.create("jev-9b") as run:
        f = run.forward([datum])
        bad(fake, {"id": f.id, "status": "completed", "result": {"outputs": "none"}})  # the future's result
        e = raises_shape_error(f.result, f"GET /v1/futures/{f.id}", "outputs")
        with pytest.raises(endor.ResponseValidationError) as again:  # settled with it
            f.result()
        assert again.value is e
        g = run.forward([datum])
        bad(fake, {"id": g.id})  # the future itself
        raises_shape_error(g.result, f"GET /v1/futures/{g.id}", "status")
