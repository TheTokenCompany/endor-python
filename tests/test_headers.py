"""Every request tells the API which SDK, version, runtime and SDK method it comes from."""

from __future__ import annotations

import logging
import uuid

import httpx
import pytest

import endor
from endor import EndorClient, cli
from endor._headers import RUNTIME, sdk_context
from endor._log import redact_headers
from endor.recipes import SupervisedConfig, supervised

from .conftest import API_KEY, BASE_URL, DEPT, rows
from .fake_api import FakeEndor, HTTPError

IDENTITY = {
    "user-agent": f"endor-python/{endor.__version__}",
    "x-endor-sdk": "endor-python",
    "x-endor-sdk-version": endor.__version__,
    "x-endor-runtime": RUNTIME,
    "x-endor-sdk-interface": "python",
    "accept": "application/json",
    "authorization": f"Bearer {API_KEY}",
}


def test_every_request_carries_identity(client: EndorClient, fake: FakeEndor) -> None:
    client.whoami()
    client.system_one("x", {"d": DEPT})
    project = client.projects.create("hdr")
    run = project.runs.create("jevk5-4b")
    run.forward(endor.data.rows_to_datums(rows(2))).result()
    run.close()
    assert len(fake.requests) >= 6
    for req in fake.requests:
        for k, v in IDENTITY.items():
            assert req.headers[k] == v, (req.path, k)
        uuid.UUID(req.headers["x-endor-client-request-id"])
        assert "x-endor-retry-count" not in req.headers
    assert RUNTIME.startswith("python/")


def test_method_header_names_the_sdk_call(client: EndorClient, fake: FakeEndor) -> None:
    client.whoami()
    client.system_one("x", {"d": DEPT})
    client.models.list()
    client.base_models()
    project = client.projects.get_or_create("m")
    project.datasets.upload("d", rows(2))
    list(project.datasets.rows("d"))
    run = project.runs.create("jevk5-4b")
    run.forward_backward(endor.data.rows_to_datums(rows(1))).result()
    run.optim_step().result()
    run.save_checkpoint("v1").result()
    run.log({"a": 1.0})
    run.metrics()
    run.log_eval("m/v1", {"accuracy": 1.0})
    project.models.get("v1")
    project.models.set_ttl("v1", 7200)
    project.models.archive_url("v1")
    project.evaluate("m/v1", "d").result()
    project.evaluations()
    run.close()
    project.models.delete("v1")
    project.datasets.delete("d")
    seen = [(r.method, r.path, r.headers["x-endor-sdk-method"]) for r in fake.requests]
    expected = [
        ("GET", "/v1/whoami", "client.whoami"),
        ("POST", "/v1/systemone", "client.system_one"),
        ("GET", "/v1/models", "client.models.list"),
        ("GET", "/v1/models", "client.base_models"),
        ("GET", "/v1/projects/m", "projects.get"),
        ("POST", "/v1/projects", "projects.create"),
        ("POST", "/v1/projects/m/datasets", "project.datasets.upload"),
        ("GET", "/v1/projects/m/datasets/d/rows", "project.datasets.rows"),
        ("POST", "/v1/projects/m/runs", "project.runs.create"),
        ("POST", "/v1/runs/run_0001/forward_backward", "run.forward_backward"),
        ("GET", "/v1/futures/fut_0002", "run.forward_backward"),
        ("POST", "/v1/runs/run_0001/optim_step", "run.optim_step"),
        ("GET", "/v1/futures/fut_0003", "run.optim_step"),
        ("POST", "/v1/runs/run_0001/save_checkpoint", "run.save_checkpoint"),
        ("GET", "/v1/futures/fut_0004", "run.save_checkpoint"),
        ("POST", "/v1/runs/run_0001/metrics", "run.log"),
        ("GET", "/v1/runs/run_0001/metrics", "run.metrics"),
        ("POST", "/v1/runs/run_0001/evaluations", "run.log_eval"),
        ("GET", "/v1/projects/m/models/v1", "project.models.get"),
        ("PATCH", "/v1/projects/m/models/v1", "project.models.set_ttl"),
        ("GET", "/v1/projects/m/models/v1/archive_url", "project.models.archive_url"),
        ("POST", "/v1/projects/m/evaluations", "project.evaluate"),
        ("GET", "/v1/futures/fut_0005", "project.evaluate"),
        ("GET", "/v1/evaluations/evl_0002", "project.evaluate"),
        ("GET", "/v1/projects/m/evaluations", "project.evaluations"),
        ("POST", "/v1/runs/run_0001/close", "run.close"),
        ("DELETE", "/v1/projects/m/models/v1", "project.models.delete"),
        ("DELETE", "/v1/projects/m/datasets/d", "project.datasets.delete"),
    ]
    assert seen == expected


def test_idempotency_key_on_creates_only(client: EndorClient, fake: FakeEndor) -> None:
    project = client.projects.create("idem")
    project.datasets.upload("d", rows(1))
    run = project.runs.create("jevk5-4b")
    run.forward(endor.data.rows_to_datums(rows(1)))
    run.log_eval("x", {})
    project.evaluate("jevk5-4b", "d")
    client.whoami()
    with_key = {(r.method, r.path) for r in fake.requests if "idempotency-key" in r.headers}
    assert with_key == {
        ("POST", "/v1/projects"),
        ("POST", "/v1/projects/idem/datasets"),
        ("POST", "/v1/projects/idem/runs"),
        ("POST", "/v1/runs/run_0001/evaluations"),
        ("POST", "/v1/projects/idem/evaluations"),
    }
    keys = [r.headers["idempotency-key"] for r in fake.requests if "idempotency-key" in r.headers]
    assert len(set(keys)) == len(keys)
    for k in keys:
        uuid.UUID(k)


def test_idempotency_key_and_request_id_survive_retries(client: EndorClient, fake: FakeEndor) -> None:
    fake.fail_next.append(httpx.ConnectError("reset"))
    fake.fail_next.append(HTTPError(503, "unavailable", "try again"))
    client.projects.create("retried")
    reqs = [r for r in fake.requests if r.path == "/v1/projects"]
    assert len(reqs) == 3
    assert len({r.headers["idempotency-key"] for r in reqs}) == 1
    assert len({r.headers["x-endor-client-request-id"] for r in reqs}) == 1
    assert [r.headers.get("x-endor-retry-count") for r in reqs] == [None, "1", "2"]


def test_idempotent_replay_returns_the_same_resource(client: EndorClient, fake: FakeEndor) -> None:
    fake.fail_next.append(httpx.ReadTimeout("slow"))  # the server may have created it before we timed out
    project = client.projects.create("once")
    assert project.name == "once" and len(fake.projects) == 1


def test_user_headers_pass_through_but_cannot_override_identity(fake: FakeEndor) -> None:
    c = EndorClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        transport=httpx.MockTransport(fake.handler),
        headers={"X-Team": "search", "X-Endor-SDK": "forged", "User-Agent": "mine"},
    )
    c.whoami()
    h = fake.requests[-1].headers
    assert (
        h["x-team"] == "search" and h["x-endor-sdk"] == "endor-python" and h["user-agent"].startswith("endor-python/")
    )


def test_cli_interface_header(client: EndorClient, fake: FakeEndor) -> None:
    assert cli.main(["whoami"], client=client) == 0
    assert fake.requests[-1].headers["x-endor-sdk-interface"] == "cli"
    client.whoami()
    assert fake.requests[-1].headers["x-endor-sdk-interface"] == "python"


def test_recipe_header(client: EndorClient, fake: FakeEndor) -> None:
    cfg = SupervisedConfig(project="rcp", batch_size=8, eval_base=False, model_name="m")
    supervised.train(cfg, rows(12), client=client)
    recipes = {r.headers.get("x-endor-sdk-recipe") for r in fake.requests}
    assert recipes == {"supervised"}
    client.whoami()
    assert "x-endor-sdk-recipe" not in fake.requests[-1].headers


def test_sdk_context_nesting() -> None:
    from endor._headers import context_headers

    with sdk_context(interface="cli"):
        with sdk_context(recipe="distill"):
            h = context_headers("x")
            assert h["X-Endor-SDK-Interface"] == "cli" and h["X-Endor-SDK-Recipe"] == "distill"
        assert "X-Endor-SDK-Recipe" not in context_headers(None)
    assert context_headers(None)["X-Endor-SDK-Interface"] == "python"


def test_secrets_are_redacted_in_debug_logs(client: EndorClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="endor"):
        client.whoami()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "headers=" in text and API_KEY not in text and "***" in text
    assert redact_headers({"Authorization": "Bearer x", "X-Api-Key": "k", "Accept": "a"}) == {
        "Authorization": "***",
        "X-Api-Key": "***",
        "Accept": "a",
    }
