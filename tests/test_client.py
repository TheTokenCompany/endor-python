from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

import endor
from endor import (
    ChoiceAnswer,
    EndorClient,
    EndorError,
    NotFoundError,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    UnprocessableEntityError,
)

from .conftest import ANGER, API_KEY, BASE_URL, DEPT, URGENT
from .fake_api import FakeEndor


class TestConstructor:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ENDOR_API_KEY", raising=False)
        with pytest.raises(EndorError, match="ENDOR_API_KEY"):
            EndorClient()

    def test_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENDOR_API_KEY", "  edk_env  ")
        monkeypatch.setenv("ENDOR_BASE_URL", "https://env.endor.test/")
        monkeypatch.setenv("ENDOR_DEFAULT_MODEL", "jev-9b")
        c = EndorClient()
        assert c.api_key == "edk_env"
        assert c.base_url == "https://env.endor.test"
        assert c.default_model == "jev-9b"
        c.close()

    def test_explicit_beats_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENDOR_API_KEY", "edk_env")
        c = EndorClient(api_key="edk_arg", base_url=BASE_URL)
        assert c.api_key == "edk_arg" and c.base_url == BASE_URL
        assert c.default_model == endor.client.DEFAULT_MODEL

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
            res = await c.system_one_async("x", {"u": URGENT}, model="jev-9b")
            assert res.nouls["u"].noul == 0.5

    def test_user_http_client_is_not_closed(self, fake: FakeEndor) -> None:
        hc = httpx.Client(transport=httpx.MockTransport(fake.handler))
        c = EndorClient(api_key=API_KEY, base_url=BASE_URL, http_client=hc)
        c.whoami()
        c.close()
        assert not hc.is_closed


class TestDecisions:
    def test_all_three_types(self, client: EndorClient) -> None:
        res = client.system_one({"body": "charged twice"}, {"dept": DEPT, "urgent": URGENT, "anger": ANGER})
        assert isinstance(res, SystemOneResponse)
        assert res.model == client.default_model
        assert isinstance(res.choices["dept"], ChoiceAnswer) and res.choices["dept"].choice in ("billing", "tech")
        assert isinstance(res.nouls["urgent"], NoulAnswer) and res.nouls["urgent"].noul == 0.5
        s = res.scores["anger"]
        assert isinstance(s, ScoreAnswer) and s.legend["0"] == "calm" and set(s.probabilities) == {"0", "1", "2"}
        assert res.usage.input_tokens is not None and res.usage.output_tokens == 0
        assert set(res.answers) == {"dept", "urgent", "anger"}

    def test_dict_questions_and_default_model(self, client: EndorClient, fake: FakeEndor) -> None:
        res = client.system_one("text", {"q": {"type": "noul", "instructions": "ok?"}})
        assert res.nouls["q"].noul == 0.5
        assert fake.requests[-1].body["model"] == client.default_model
        assert fake.requests[-1].body["questions"]["q"] == {"type": "noul", "instructions": "ok?"}

    def test_question_wire_form_omits_unset(self, client: EndorClient, fake: FakeEndor) -> None:
        client.system_one("t", {"d": endor.Choice(criteria={"a": None})})
        assert fake.requests[-1].body["questions"]["d"] == {"type": "choice", "criteria": {"a": None}}

    def test_response_model(self, client: EndorClient) -> None:
        class Typed(SystemOneResponse):
            dept: ChoiceAnswer
            urgent: NoulAnswer

        res = client.system_one("x", {"dept": DEPT, "urgent": URGENT}, response_model=Typed)
        assert isinstance(res, Typed) and res.dept.choice in ("billing", "tech") and res.urgent.noul == 0.5

    def test_response_model_mismatch(self, client: EndorClient) -> None:
        class Typed(SystemOneResponse):
            missing: ChoiceAnswer

        with pytest.raises(endor.ResponseValidationError, match="missing"):
            client.system_one("x", {"dept": DEPT}, response_model=Typed)

    def test_extra_body_and_headers(self, client: EndorClient, fake: FakeEndor) -> None:
        client.system_one("x", {"d": DEPT}, extra_body={"trace": "abc"}, extra_headers={"X-Trace": "1"})
        req = fake.requests[-1]
        assert req.body["trace"] == "abc" and req.headers["x-trace"] == "1"

    def test_project_model_id(self, client: EndorClient, project: endor.Project) -> None:
        with project.runs.create("pplx-decider-v1.1-27b") as run:
            model = run.save_checkpoint("v1").result()
        assert client.system_one("x", {"d": DEPT}, model=model).model == f"{project.name}/v1"

    def test_unknown_model_is_404(self, client: EndorClient) -> None:
        with pytest.raises(NotFoundError) as e:
            client.system_one("x", {"d": DEPT}, model="nope/none")
        assert e.value.code == "unknown_model"

    def test_too_many_options_names_the_question(self, client: EndorClient) -> None:
        wide = endor.Choice(criteria={f"o{i}": None for i in range(20)})
        with pytest.raises(UnprocessableEntityError) as e:
            client.system_one("x", {"wide": wide}, model="jev-9b")
        assert e.value.code == "invalid_options" and "questions.wide" in str(e.value)

    def test_local_validation(self, client: EndorClient) -> None:
        with pytest.raises(ValueError, match="at least one question"):
            client.system_one("x", {})
        with pytest.raises(ValueError, match="at most 64"):
            client.system_one("x", {f"q{i}": URGENT for i in range(65)})
        with pytest.raises(ValueError, match="criteria"):
            client.system_one("x", {"q": {"type": "choice"}})
        with pytest.raises(TypeError):
            client.system_one("x", {"q": 42})  # type: ignore[dict-item]

    async def test_async(self, client: EndorClient) -> None:
        res = await client.system_one_async("x", {"u": URGENT}, model="decider-2b")
        assert res.nouls["u"].noul == 0.5
        models = await client.models.list_async()
        assert any(m.name == "decider-2b" for m in models.models)


class TestCatalogAndAccount:
    def test_models_list_keeps_endor_metadata(self, client: EndorClient, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.save_checkpoint("v1").result()
        listed = client.models.list()
        by_name = {m.name: m for m in listed.models}
        assert by_name["jev-9b"].kind == "base" and by_name["jev-9b"].endor["max_options"] == 16
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

    def test_whoami(self, client: EndorClient) -> None:
        me = client.whoami()
        assert me.org_id and me.key_id and me.user_id is None  # an org API key has no user
        assert me.key_prefix and me.key_prefix.startswith("edk_")

    def test_usage_rows(self, client: EndorClient) -> None:
        now = datetime.now(timezone.utc)
        for u in client.usage(now - timedelta(days=1), now + timedelta(hours=1)):
            assert u.kind in ("decide", "train")
            assert (u.input_tokens is None) == (u.kind == "train") and (u.gpu_seconds is None) == (u.kind == "decide")

    def test_usage(self, client: EndorClient, fake: FakeEndor) -> None:
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        decide, train = client.usage(start, start + timedelta(days=3), project="tickets")
        assert decide.kind == "decide" and decide.input_tokens == 120 and decide.gpu_seconds is None
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
