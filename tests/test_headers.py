"""Every request tells the API which SDK version and runtime (User-Agent) and which SDK method it comes from."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import httpx
import pytest

import endor
from endor import EndorClient, cli
from endor._headers import RUNTIME, RUNTIME_HEADER, sdk_context, user_agent
from endor._log import redact_headers
from endor.recipes import SupervisedConfig, supervised

from .conftest import API_KEY, BASE_URL, DEPT, rows, unique
from .fake_api import FakeEndor, HTTPError

UA = f"endor-python/{endor.__version__} ({RUNTIME})"
IDENTITY = {
    "user-agent": UA,
    "x-endor-sdk": "endor-python",
    "x-endor-sdk-version": endor.__version__,
    "x-endor-runtime": RUNTIME_HEADER,
    "x-endor-sdk-interface": "python",
    "accept": "application/json",
    "authorization": f"Bearer {API_KEY}",
}


def test_every_request_carries_identity(client: EndorClient, fake: FakeEndor) -> None:
    client.whoami()
    project = client.projects.create(unique("hdr"), base_model="jev-9b")
    client.system_one("x", {"d": DEPT}, model=project.ref)
    with client.runs.create("jev-9b") as run:
        run.forward(endor.data.rows_to_datums(rows(2))).result()
    assert len(fake.requests) >= 6
    for req in fake.requests:
        for k, v in IDENTITY.items():
            assert req.headers[k] == v, (req.path, k)
        uuid.UUID(req.headers["x-endor-client-request-id"])
        assert "x-endor-retry-count" not in req.headers
    assert RUNTIME.startswith("python 3.") and RUNTIME.count(";") == 2
    assert RUNTIME_HEADER.startswith("python/3.") and RUNTIME_HEADER.endswith(")")


def test_method_header_names_the_sdk_call(client: EndorClient, fake: FakeEndor, tmp_path: Path) -> None:
    client.whoami()
    name = unique("m")
    fake.create_project({"name": name, "base_model": "jev-9b"})  # in the fake directly: no request
    fake.add_version(name)
    client.system_one("x", {"d": DEPT}, model=name)
    client.models.list()
    client.base_models()
    project = client.projects.get_or_create(name, base_model="jev-9b")
    project.versions()
    project.promote("v1")
    project.set_auto_promote(True)
    client.datasets.upload("d", rows(2))
    list(client.datasets.rows("d"))
    run = client.custom_models.create("cm", base_model="jev-9b")
    run.forward_backward(endor.data.rows_to_datums(rows(1))).result()
    run.optim_step().result()
    run.save_checkpoint().result()
    run.log({"a": 1.0})
    run.metrics()
    run.log_eval("cm", {"accuracy": 1.0})
    client.custom_models.get("cm")
    client.custom_models.set_ttl("cm", 7200)
    client.custom_models.download("cm", tmp_path)
    client.custom_models.promote("cm")
    client.custom_models.archive("cm")
    client.custom_models.list()
    client.evaluations.create("cm", "d").result()
    client.evaluations.list()
    run.close()
    client.custom_models.delete("cm")
    client.datasets.delete("d")
    seen = [(r.method, r.path, r.headers["x-endor-sdk-method"]) for r in fake.requests]
    expected = [
        ("GET", "/v1/whoami", "client.whoami"),
        ("POST", "/v1/systemone", "client.system_one"),
        ("GET", "/v1/models", "client.models.list"),
        ("GET", "/v1/base_models", "client.base_models"),
        ("GET", f"/v1/projects/{name}", "projects.get"),
        ("GET", f"/v1/projects/{name}/versions", "project.versions"),
        ("POST", f"/v1/projects/{name}/versions/v1/promote", "project.promote"),
        ("PATCH", f"/v1/projects/{name}", "project.set_auto_promote"),
        ("POST", "/v1/datasets", "datasets.upload"),
        ("GET", "/v1/datasets/d/rows", "datasets.rows"),
        ("POST", "/v1/runs", "runs.create"),
        ("POST", "/v1/runs/run_0001/forward_backward", "run.forward_backward"),
        ("GET", "/v1/futures/fut_0002", "run.forward_backward"),
        ("POST", "/v1/runs/run_0001/optim_step", "run.optim_step"),
        ("GET", "/v1/futures/fut_0003", "run.optim_step"),
        ("POST", "/v1/runs/run_0001/save_checkpoint", "run.save_checkpoint"),
        ("GET", "/v1/futures/fut_0004", "run.save_checkpoint"),
        ("POST", "/v1/runs/run_0001/metrics", "run.log"),
        ("GET", "/v1/runs/run_0001/metrics", "run.metrics"),
        ("POST", "/v1/runs/run_0001/evaluations", "run.log_eval"),
        ("GET", "/v1/custom_models/cm", "custom_models.get"),
        ("PATCH", "/v1/custom_models/cm", "custom_models.set_ttl"),
        ("GET", "/v1/custom_models/cm/download", "custom_models.download"),
        ("POST", "/v1/custom_models/cm/promote", "custom_models.promote"),
        ("POST", "/v1/custom_models/cm/demote", "custom_models.archive"),
        ("GET", "/v1/custom_models", "custom_models.list"),
        ("POST", "/v1/evaluations", "evaluations.create"),
        ("GET", "/v1/futures/fut_0005", "evaluations.create"),
        ("GET", "/v1/evaluations/evl_0002", "evaluations.get"),
        ("GET", "/v1/evaluations", "evaluations.list"),
        ("POST", "/v1/runs/run_0001/close", "run.close"),
        ("DELETE", "/v1/custom_models/cm", "custom_models.delete"),
        ("DELETE", "/v1/datasets/d", "datasets.delete"),
    ]
    assert seen == expected


def test_idempotency_key_on_creates_only(client: EndorClient, fake: FakeEndor) -> None:
    name = unique("idem")
    project = client.projects.create(name, base_model="jev-9b")
    client.datasets.upload("d", rows(1))
    with client.runs.create("jev-9b") as run:
        run.forward(endor.data.rows_to_datums(rows(1)))
        run.log_eval("x", {})
    client.evaluations.create("jev-9b", "d")
    project.set_auto_promote(False)
    client.whoami()
    with_key = {(r.method, r.path) for r in fake.requests if "idempotency-key" in r.headers}
    assert with_key == {
        ("POST", "/v1/projects"),
        ("POST", "/v1/datasets"),
        ("POST", "/v1/runs"),
        ("POST", "/v1/runs/run_0001/evaluations"),
        ("POST", "/v1/evaluations"),
    }
    keys = [r.headers["idempotency-key"] for r in fake.requests if "idempotency-key" in r.headers]
    assert len(set(keys)) == len(keys)
    for k in keys:
        uuid.UUID(k)


def test_idempotency_key_and_request_id_survive_retries(client: EndorClient, fake: FakeEndor) -> None:
    fake.fail_next.append(httpx.ConnectError("reset"))
    fake.fail_next.append(HTTPError(503, "unavailable", "try again"))
    client.projects.create(unique("retried"), base_model="jev-9b")
    reqs = [r for r in fake.requests if r.path == "/v1/projects"]
    assert len(reqs) == 3
    assert len({r.headers["idempotency-key"] for r in reqs}) == 1
    assert len({r.headers["x-endor-client-request-id"] for r in reqs}) == 1
    assert [r.headers.get("x-endor-retry-count") for r in reqs] == [None, "1", "2"]


def test_idempotent_replay_returns_the_same_resource(client: EndorClient, fake: FakeEndor) -> None:
    fake.fail_next.append(httpx.ReadTimeout("slow"))  # the server may have created it before we timed out
    name = unique("once")
    project = client.projects.create(name, base_model="jev-9b")
    assert project.name == name and len(fake.projects) == 1


def test_user_headers_pass_through_but_cannot_override_identity(fake: FakeEndor) -> None:
    c = EndorClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        transport=httpx.MockTransport(fake.handler),
        headers={"X-Team": "search", "X-Endor-SDK": "forged", "X-Endor-SDK-Method": "forged", "User-Agent": "mine"},
    )
    c.whoami()
    h = fake.requests[-1].headers
    assert h["x-team"] == "search" and h["x-endor-sdk-method"] == "client.whoami" and h["user-agent"] == UA
    assert h["x-endor-sdk"] == "endor-python"


def test_cli_is_named_in_the_user_agent(client: EndorClient, fake: FakeEndor) -> None:
    assert cli.main(["whoami"], client=client) == 0
    h = fake.requests[-1].headers
    assert (
        h["user-agent"] == f"endor-python/{endor.__version__} ({RUNTIME}; cli)" and h["x-endor-sdk-interface"] == "cli"
    )
    client.whoami()
    h = fake.requests[-1].headers
    assert h["user-agent"] == UA and h["x-endor-sdk-interface"] == "python"


def test_recipe_header(client: EndorClient, fake: FakeEndor) -> None:
    cfg = SupervisedConfig(batch_size=8, eval_base=False, model_name="m")
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
            assert h == {"X-Endor-SDK-Method": "x", "X-Endor-SDK-Recipe": "distill"}
            assert user_agent().endswith("; cli)")
        assert "X-Endor-SDK-Recipe" not in context_headers(None)
    assert context_headers(None) == {} and not user_agent().endswith("; cli)")


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
