"""Reports to the Endor team: endor.report and client.report, the hint at the end of every error, the CLI command."""

from __future__ import annotations

import json

import httpx
import pytest

import endor
from endor import EndorClient, NotFoundError, RateLimitError, RetryPolicy, cli
from endor._http import LAST

from .conftest import BASE_URL
from .fake_api import FakeEndor, HTTPError


@pytest.fixture(autouse=True)
def _fresh_last() -> None:
    LAST.request_id = LAST.failed_request_id = LAST.error = None


def make_client(fake: FakeEndor) -> EndorClient:
    return EndorClient(
        api_key="edk_test",
        base_url=BASE_URL,
        transport=httpx.MockTransport(fake.handler),
        async_transport=httpx.MockTransport(fake.handler),
        retry=RetryPolicy(backoff_initial=0.0, backoff_max=0.0),
    )


def test_client_report_names_the_org_and_the_failed_request(fake: FakeEndor) -> None:
    c = make_client(fake)
    with pytest.raises(NotFoundError) as e:
        c.projects.get("nope")
    c.whoami()  # a later success: the report is still about the failure
    receipt = c.report("projects.get said not_found for a project I just made", kind="bug")
    assert receipt.id.startswith("rep_") and "Thanks" in receipt.message
    [rep] = fake.reports
    assert rep["org_id"] == "org_test" and rep["kind"] == "bug" and rep["request_id"] == e.value.request_id
    ctx = rep["context"]
    assert ctx["sdk"] == "endor-python" and ctx["sdk_version"] == endor.__version__ and ctx["python"]
    assert (ctx["last_error_status"], ctx["last_error_code"]) == (404, "not_found")
    assert ctx["last_error_endpoint"] == "GET /v1/projects/nope" and ctx["last_error_sdk_method"] == "projects.get"
    assert "edk_test" not in json.dumps(rep)
    req = [r for r in fake.requests if r.path == "/v1/reports"][0]
    assert req.headers["x-endor-sdk-method"] == "client.report" and req.headers.get("idempotency-key")
    # the report itself is not what the next report is about
    c.report("still broken")
    assert fake.reports[1]["request_id"] == e.value.request_id


def test_works_report_is_about_the_latest_request_and_no_failure(fake: FakeEndor) -> None:
    c = make_client(fake)
    with pytest.raises(NotFoundError):
        c.projects.get("nope")
    c.whoami()
    c.report("the integration runs", kind="works", context={"decisions_per_hour": 120, "nested": {"a": 1}})
    rep = fake.reports[0]
    assert rep["request_id"] == c._t.last.request_id and "last_error_code" not in rep["context"]
    assert rep["context"]["decisions_per_hour"] == 120 and rep["context"]["nested"] == "{'a': 1}"


def test_endor_report_without_a_key_is_anonymous(fake: FakeEndor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENDOR_API_KEY", raising=False)
    c = make_client(fake)
    with pytest.raises(NotFoundError) as e:
        c.projects.get("nope")
    receipt = endor.report(
        "I can't find how to add a base model",
        kind="confusing",
        base_url=BASE_URL,
        transport=httpx.MockTransport(fake.handler),
    )
    assert receipt.id.startswith("rep_")
    rep = fake.reports[0]
    assert rep["org_id"] is None and rep["kind"] == "confusing"
    assert rep["request_id"] == e.value.request_id  # any client's latest failure in this process
    req = [r for r in fake.requests if r.path == "/v1/reports"][0]
    assert "authorization" not in req.headers
    # a bad key is no reason to fail: the API takes the report anonymously
    endor.report("x", api_key="edk_revoked", base_url=BASE_URL, transport=httpx.MockTransport(fake.handler))
    assert fake.reports[1]["org_id"] is None


async def test_async_reports(fake: FakeEndor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENDOR_API_KEY", "edk_test")
    c = make_client(fake)
    assert (await c.report_async("async works", kind="works")).id.startswith("rep_")
    r = await endor.report_async("module async", base_url=BASE_URL, async_transport=httpx.MockTransport(fake.handler))
    assert r.id.startswith("rep_") and [x["org_id"] for x in fake.reports] == ["org_test", "org_test"]


def test_messages_and_context_are_checked_and_trimmed(fake: FakeEndor) -> None:
    c = make_client(fake)
    with pytest.raises(ValueError, match="kind must be one of"):
        c.report("x", kind="nice")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="message is empty"):
        c.report("   ")
    c.report("y" * 5000, context={f"k{i}": "v" * 900 for i in range(40)})
    rep = fake.reports[0]
    assert len(rep["message"]) == 4000 and rep["message"].endswith("…")
    assert len(rep["context"]) == 30 and all(len(v) <= 500 for v in rep["context"].values() if isinstance(v, str))


def test_rate_limited_report_is_a_rate_limit_error_never_retried(fake: FakeEndor) -> None:
    c = make_client(fake)
    fake.fail_next.append(
        HTTPError(429, "rate_limited", "Thanks, we have your reports.", headers={"Retry-After": "60"})
    )
    with pytest.raises(RateLimitError) as e:
        c.report("again")
    assert e.value.retry_after == 60 and len([r for r in fake.requests if r.path == "/v1/reports"]) == 1
    assert "endor.report" not in str(e.value)  # no hint to report a failed report


def test_every_error_ends_with_how_to_report_it(fake: FakeEndor) -> None:
    c = make_client(fake)
    with pytest.raises(NotFoundError) as e:
        c.projects.get("nope")
    first, hint = str(e.value).split("\n")
    assert first.startswith("GET /v1/projects/nope: 404 not_found") and e.value.message
    assert hint == (
        "If Endor did something you didn't expect, or this message is unclear, tell the Endor team: "
        f'endor.report("what you did, what happened, what you expected", kind="bug", request_id="{e.value.request_id}")'
    )
    assert "\n" not in repr(e.value) and "endor.report" not in e.value.message
    with pytest.raises(endor.EndorError) as e2:
        c.system_one("x", {"q": endor.Noul()})  # no model: a local error, still with the hint
    assert str(e2.value).endswith('endor.report("what you did, what happened, what you expected", kind="bug")')


def test_cli_report(capsys: pytest.CaptureFixture[str], fake: FakeEndor, monkeypatch: pytest.MonkeyPatch) -> None:
    c = make_client(fake)
    assert cli.main(["-f", "json", "report", "the docs skip auth", "--kind", "confusing"], client=c) == 0
    assert json.loads(capsys.readouterr().out)["id"].startswith("rep_")
    assert fake.reports[0]["kind"] == "confusing" and fake.reports[0]["context"]["interface"] == "cli"
    # without a key (and without a client): anonymous, never "no API key"
    monkeypatch.delenv("ENDOR_API_KEY", raising=False)
    real = endor._report.report

    def through_fake(message: str, **kw: object) -> endor.ReportReceipt:
        return real(message, base_url=BASE_URL, transport=httpx.MockTransport(fake.handler), **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(endor._report, "report", through_fake)
    assert cli.main(["report", "no key yet", "--request-id", "req_1"]) == 0
    assert fake.reports[1]["org_id"] is None and fake.reports[1]["request_id"] == "req_1"
    # other commands' errors end with the CLI's way to report
    assert cli.main(["projects", "delete", "nope"], client=c) == 1
    err = capsys.readouterr().err
    assert 'endor report "what you did, what happened, what you expected" --kind bug --request-id req_' in err
