from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

import endor
from endor import (
    ChoiceAnswer,
    EndorClient,
    EndorError,
    ModelRequiresProjectError,
    NoBaseModelError,
    NotFoundError,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    UnprocessableEntityError,
)

from .conftest import ANGER, API_KEY, BASE_URL, DECIDE_BASE, DEPT, URGENT, unique
from .fake_api import FakeEndor, HTTPError


class TestConstructor:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENDOR_API_KEY", raising=False)
        with pytest.raises(EndorError, match="ENDOR_API_KEY"):
            EndorClient()

    def test_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENDOR_API_KEY", "  edk_env  ")
        monkeypatch.setenv("ENDOR_BASE_URL", "https://env.endor.test/")
        monkeypatch.setenv("ENDOR_DEFAULT_MODEL", "tickets")
        c = EndorClient()
        assert c.api_key == "edk_env"
        assert c.base_url == "https://env.endor.test"
        assert c.default_model == "tickets"
        c.close()

    def test_explicit_beats_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENDOR_API_KEY", "edk_env")
        monkeypatch.setenv("ENDOR_DEFAULT_MODEL", "tickets")
        c = EndorClient(api_key="edk_arg", base_url=BASE_URL, model="tickets/base")
        assert c.api_key == "edk_arg" and c.base_url == BASE_URL
        assert c.default_model == "tickets/base"

    def test_no_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENDOR_DEFAULT_MODEL", raising=False)
        c = EndorClient(api_key="edk_x")
        assert c.default_model is None and not hasattr(endor.client, "DEFAULT_MODEL")

    def test_rejects_bad_key_and_timeout(self) -> None:
        with pytest.raises(EndorError):
            EndorClient(api_key="edk with space")
        with pytest.raises(ValueError):
            EndorClient(api_key="edk_x", timeout=0)

    def test_transport_and_http_client_exclusive(self) -> None:
        with pytest.raises(ValueError):
            EndorClient(
                api_key="edk_x",
                transport=httpx.MockTransport(lambda r: httpx.Response(200)),
                http_client=httpx.Client(),
            )

    def test_timeout_is_kept_as_given(self) -> None:
        c = EndorClient(api_key="edk_x", timeout=5)
        assert c.timeout == 5 and c._t.timeout == 5.0

    def test_context_managers(self, fake: FakeEndor) -> None:
        with EndorClient(api_key=API_KEY, base_url=BASE_URL, transport=httpx.MockTransport(fake.handler)) as c:
            assert c.whoami().org_id == "org_test"
        assert "EndorClient" in repr(c)

    async def test_async_context_manager(self, fake: FakeEndor) -> None:
        async with EndorClient(
            api_key=API_KEY, base_url=BASE_URL, async_transport=httpx.MockTransport(fake.handler)
        ) as c:
            name = fake.create_project({"name": unique("acm"), "base_model": DECIDE_BASE})["name"]
            res = await c.system_one_async("x", {"u": URGENT}, model=name)
            assert res.nouls["u"].noul == 0.5

    def test_user_http_client_is_not_closed(self, fake: FakeEndor) -> None:
        hc = httpx.Client(transport=httpx.MockTransport(fake.handler))
        c = EndorClient(api_key=API_KEY, base_url=BASE_URL, http_client=hc)
        c.whoami()
        c.close()
        assert not hc.is_closed


class TestDecisions:
    def test_all_three_types(self, client: EndorClient, decider: str) -> None:
        res = client.system_one(
            {"body": "charged twice"}, {"dept": DEPT, "urgent": URGENT, "anger": ANGER}, model=decider
        )
        assert isinstance(res, SystemOneResponse)
        assert res.model == f"{decider}/base"  # the model that answered: a new project's live model is its base
        assert isinstance(res.choices["dept"], ChoiceAnswer) and res.choices["dept"].choice in ("billing", "tech")
        assert isinstance(res.nouls["urgent"], NoulAnswer) and res.nouls["urgent"].noul == 0.5
        s = res.scores["anger"]
        assert isinstance(s, ScoreAnswer) and s.legend["0"] == "calm" and set(s.probabilities) == {"0", "1", "2"}
        assert res.usage.input_tokens is not None and res.usage.output_tokens == 0
        assert set(res.answers) == {"dept", "urgent", "anger"}

    def test_dict_questions_and_default_model(self, client: EndorClient, fake: FakeEndor, decider: str) -> None:
        client.default_model = decider
        res = client.system_one("text", {"q": {"type": "noul", "instructions": "ok?"}})
        assert res.nouls["q"].noul == 0.5
        assert fake.requests[-1].body["model"] == decider
        assert fake.requests[-1].body["questions"]["q"] == {"type": "noul", "instructions": "ok?"}

    def test_question_wire_form_omits_unset(self, client: EndorClient, fake: FakeEndor, decider: str) -> None:
        client.system_one("t", {"d": endor.Choice(criteria={"a": None})}, model=decider)
        assert fake.requests[-1].body["questions"]["d"] == {"type": "choice", "criteria": {"a": None}}

    def test_response_model(self, client: EndorClient, decider: str) -> None:
        class Typed(SystemOneResponse):
            dept: ChoiceAnswer
            urgent: NoulAnswer

        res = client.system_one("x", {"dept": DEPT, "urgent": URGENT}, model=decider, response_model=Typed)
        assert isinstance(res, Typed) and res.dept.choice in ("billing", "tech") and res.urgent.noul == 0.5

    def test_response_model_mismatch(self, client: EndorClient, decider: str) -> None:
        class Typed(SystemOneResponse):
            missing: ChoiceAnswer

        with pytest.raises(endor.ResponseValidationError, match="missing"):
            client.system_one("x", {"dept": DEPT}, model=decider, response_model=Typed)

    def test_extra_body_and_headers(self, client: EndorClient, fake: FakeEndor, decider: str) -> None:
        client.system_one(
            "x", {"d": DEPT}, extra_body={"trace": "abc", "model": decider}, extra_headers={"X-Trace": "1"}
        )
        req = fake.requests[-1]
        assert req.body["trace"] == "abc" and req.body["model"] == decider and req.headers["x-trace"] == "1"

    def test_project_model_id(self, client: EndorClient, project: endor.Project) -> None:
        with project.runs.create() as run:
            model = run.save_checkpoint("v1").result()
        assert client.system_one("x", {"d": DEPT}, model=model).model == f"{project.name}/v1"

    def test_project_base_and_live_model(self, client: EndorClient, fake: FakeEndor, project: endor.Project) -> None:
        with project.runs.create() as run:  # on the project's base model
            run.save_checkpoint("v1").result()
        assert client.system_one("x", {"d": DEPT}, model=project.name).model == f"{project.name}/base"
        assert client.system_one("x", {"d": DEPT}, model=f"{project.name}/base").model == f"{project.name}/base"
        fake.promote(project.name, "v1")
        assert client.system_one("x", {"d": DEPT}, model=project.name).model == f"{project.name}/v1"

    def test_no_model_fails_before_the_call(self, client: EndorClient, fake: FakeEndor) -> None:
        assert client.default_model is None
        with pytest.raises(EndorError, match="no model") as e:
            client.system_one("x", {"d": DEPT})
        assert "<project>/base" in str(e.value) and "ENDOR_DEFAULT_MODEL" in str(e.value)
        assert not fake.requests

    async def test_no_model_fails_before_the_call_async(self, client: EndorClient, fake: FakeEndor) -> None:
        with pytest.raises(EndorError, match="no model"):
            await client.system_one_async("x", {"d": DEPT})
        assert not fake.requests

    def test_bare_base_model_id_needs_a_project(self, client: EndorClient) -> None:
        with pytest.raises(ModelRequiresProjectError) as e:
            client.system_one("x", {"d": DEPT}, model="decider-2b")
        assert isinstance(e.value, UnprocessableEntityError)
        assert e.value.status == 422 and e.value.code == "model_requires_project" and e.value.param == "model"

    def test_project_without_base_model(self, client: EndorClient, fake: FakeEndor, project: endor.Project) -> None:
        fake.projects[project.name]["base_model"] = None  # only projects made before base models were required
        with pytest.raises(NoBaseModelError) as e:
            client.system_one("x", {"d": DEPT}, model=project.name)
        assert isinstance(e.value, endor.ConflictError) and e.value.status == 409 and e.value.code == "no_base_model"
        with pytest.raises(NoBaseModelError):
            client.system_one("x", {"d": DEPT}, model=f"{project.name}/base")

    def test_unknown_model_is_404(self, client: EndorClient) -> None:
        with pytest.raises(NotFoundError) as e:
            client.system_one("x", {"d": DEPT}, model="nope/none")
        assert e.value.code == "unknown_model"

    def test_too_many_options_names_the_question(self, client: EndorClient, decider: str) -> None:
        wide = endor.Choice(criteria={f"o{i}": None for i in range(20)})  # the base, jev-9b, reads 16
        with pytest.raises(UnprocessableEntityError) as e:
            client.system_one("x", {"wide": wide}, model=decider)
        assert e.value.code == "invalid_options" and "questions.wide" in str(e.value)

    def test_local_validation(self, client: EndorClient) -> None:
        with pytest.raises(ValueError, match="at least one question"):
            client.system_one("x", {}, model="t")
        with pytest.raises(ValueError, match="at most 64"):
            client.system_one("x", {f"q{i}": URGENT for i in range(65)}, model="t")
        with pytest.raises(ValueError, match="criteria"):
            client.system_one("x", {"q": {"type": "choice"}}, model="t")
        with pytest.raises(TypeError):
            client.system_one("x", {"q": 42}, model="t")  # type: ignore[dict-item]

    async def test_async(self, client: EndorClient, decider: str) -> None:
        res = await client.system_one_async("x", {"u": URGENT}, model=decider)
        assert res.nouls["u"].noul == 0.5
        models = await client.models.list_async()
        assert any(m.name == decider and m.kind == "live" for m in models.models)


class TestCatalogAndAccount:
    def test_models_list_keeps_endor_metadata(self, client: EndorClient, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.save_checkpoint("v1").result()
        listed = client.models.list()
        by_name = {m.name: m for m in listed.models}
        assert "jev-9b" not in by_name  # a bare base model id is not a name a decision can send
        assert by_name[project.name].kind == "live" and by_name[project.name].endor["model"] == f"{project.name}/base"
        assert by_name[f"{project.name}/base"].kind == "base"
        saved = by_name[f"{project.name}/v1"]
        assert saved.kind == "model" and saved.endor["base_model"] == "jev-9b" and saved.release_date

    def test_base_models(self, client: EndorClient, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.save_checkpoint("v1").result()
        bases = client.base_models()
        assert {b.id for b in bases} == {"pplx-decider-v1.1-27b", "gev-26b", "jev-9b", "decider-2b", "gliner2.5-decide"}
        assert next(b for b in bases if b.id == "jev-9b").max_options == 16
        assert all(hasattr(b, "price_per_gpu_hour") and hasattr(b, "price_per_mtok_decide") for b in bases)

    def test_base_model_prices(self, client: EndorClient, fake: FakeEndor) -> None:
        jevk = next(b for b in client.base_models() if b.id == "jev-9b")
        assert jevk.price_per_gpu_hour == 3.0 and jevk.price_per_mtok_decide == 0.5
        assert jevk.price_per_mtok_decide_continuous_learning == 0.75
        assert jevk.hf_repo == "org/jev-9b" and jevk.hf_revision and jevk.contract and jevk.params

    def test_base_models_from_an_older_api(self, client: EndorClient, fake: FakeEndor) -> None:
        """An API without /v1/base_models listed the catalog in /v1/models."""
        old = {
            "name": "jev-9b",
            "description": "",
            "release_date": "2026-10-01",
            "endor": {"kind": "base", "id": "jev-9b", "max_options": 16, "price_per_mtok_decide": 0.1},
        }
        real = fake.route

        def route(method: str, path: str, body: Any, params: dict[str, str]) -> tuple[int, Any]:
            if path == "/v1/base_models":
                raise HTTPError(404, "not_found", "no route")
            if path == "/v1/models":
                return 200, {"models": [old]}
            return real(method, path, body, params)

        fake.route = route  # type: ignore[method-assign]
        assert [(b.id, b.max_options) for b in client.base_models()] == [("jev-9b", 16)]

    def test_whoami(self, client: EndorClient) -> None:
        me = client.whoami()
        assert me.org_id and me.key_id and me.user_id is None  # an org API key has no user
        assert me.key_prefix and me.key_prefix.startswith("edk_")
        assert me.limits["max_projects_per_org"] >= 1 and me.limits["max_active_runs"] >= 1

    def test_usage_rows(self, client: EndorClient) -> None:
        now = datetime.now(timezone.utc)
        for u in client.usage(now - timedelta(days=1), now + timedelta(hours=1)):
            assert u.kind in ("decide", "train")
            assert (u.input_tokens is None) == (u.kind == "train") and (u.gpu_seconds is None) == (u.kind == "decide")

    def test_usage(self, client: EndorClient, fake: FakeEndor) -> None:
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        decide, train = client.usage(start, start + timedelta(days=3), project="tickets")
        assert decide.kind == "decide" and decide.input_tokens == 120 and decide.gpu_seconds is None
        assert decide.continuous_learning is True and decide.price_per_mtok == 0.3
        assert decide.base_cost_usd == 0.000024 and decide.continuous_learning_cost_usd == 0.000012
        assert train.continuous_learning is None and train.price_per_mtok is None and train.base_cost_usd is None
        assert train.kind == "train" and train.gpu_seconds == 360 and train.training_run_id == "run_0001"
        assert fake.requests[-1].params == {
            "starting_on": start.isoformat(),
            "ending_before": (start + timedelta(days=3)).isoformat(),
            "project": "tickets",
        }
        with pytest.raises(UnprocessableEntityError) as e:
            client.usage(start, start + timedelta(days=20))
        assert e.value.param == "ending_before"
        client.usage(datetime(2026, 10, 1), datetime(2026, 10, 2))  # no timezone: taken as UTC
        assert fake.requests[-1].params["starting_on"] == "2026-10-01T00:00:00+00:00"
