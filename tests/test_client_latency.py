"""Each request's end-to-end time (sent to response read), reported to the API on the next request as
``X-Endor-Client-Latency: <request id>=<ms>``: only the latest, at most once, sync and async; never an error, and off
with ``report_latency=False`` or ``ENDOR_REPORT_LATENCY=0``."""

from __future__ import annotations

import asyncio
import re
import time

import httpx
import pytest

from endor import EndorClient, InternalServerError, RetryPolicy

from .conftest import API_KEY, BASE_URL
from .fake_api import FakeEndor, HTTPError

HEADER = "x-endor-client-latency"


def make_client(fake: FakeEndor, handler=None, **kw: object) -> EndorClient:  # type: ignore[no-untyped-def]
    h = handler or fake.handler
    kw.setdefault("retry", RetryPolicy(backoff_initial=0.0, backoff_max=0.0))
    return EndorClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        transport=httpx.MockTransport(h),
        async_transport=httpx.MockTransport(h),
        **kw,  # type: ignore[arg-type]
    )


def reported(fake: FakeEndor) -> list[str | None]:
    return [r.headers.get(HEADER) for r in fake.requests]


def ms_of(value: str | None, request_id: str) -> int:
    assert value is not None
    m = re.fullmatch(re.escape(request_id) + r"=(\d+)", value)
    assert m, value
    return int(m[1])


def test_the_next_request_reports_the_one_before(fake: FakeEndor) -> None:
    c = make_client(fake)
    ids = []
    for _ in range(3):
        c.whoami()
        ids.append(c._t.last.request_id)
    got = reported(fake)
    assert got[0] is None  # nothing before it
    assert ms_of(got[1], ids[0]) >= 0 and ms_of(got[2], ids[1]) >= 0  # each once, only the latest


def test_the_time_is_the_whole_call(fake: FakeEndor) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        time.sleep(0.05)
        return fake.handler(request)

    c = make_client(fake, slow)
    c.whoami()
    first = c._t.last.request_id
    c.whoami()
    assert ms_of(reported(fake)[1], first) >= 50


async def test_async_calls_report_too(fake: FakeEndor) -> None:
    c = make_client(fake)
    await c.models.list_async()
    first = c._t.last.request_id
    await c.models.list_async()
    assert reported(fake)[0] is None and ms_of(reported(fake)[1], first) >= 0
    await asyncio.gather(c.models.list_async(), c.models.list_async())  # concurrent calls: still at most once each
    values = [v for v in reported(fake) if v]
    assert len(values) == len(set(values))


def test_errors_are_timed_and_a_retry_carries_the_report(fake: FakeEndor) -> None:
    c = make_client(fake)
    fake.fail_next.append(HTTPError(500, "internal", "bug"))
    c.whoami()  # a 500, then its retry
    got = reported(fake)
    assert got[0] is None and got[1] is not None and got[1].startswith("req_")
    c2 = make_client(fake, retry=RetryPolicy(max_retries=0))
    fake.fail_next.append(HTTPError(500, "internal", "bug"))
    with pytest.raises(InternalServerError) as e:
        c2.whoami()
    c2.whoami()
    assert ms_of(reported(fake)[-1], str(e.value.request_id)) >= 0


@pytest.mark.parametrize("off", ["arg", "env"])
def test_off(fake: FakeEndor, monkeypatch: pytest.MonkeyPatch, off: str) -> None:
    if off == "env":
        monkeypatch.setenv("ENDOR_REPORT_LATENCY", "0")
        c = make_client(fake)
    else:
        c = make_client(fake, report_latency=False)
    c.whoami()
    c.whoami()
    assert reported(fake) == [None, None]


def test_the_argument_wins_over_the_environment(fake: FakeEndor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENDOR_REPORT_LATENCY", "false")
    c = make_client(fake, report_latency=True)
    c.whoami()
    c.whoami()
    assert reported(fake)[1] is not None


@pytest.mark.parametrize("request_id", [None, "", "req_ü", "req_a b", "req_" + "a" * 37, "x" * 300, "abc=1"])
def test_odd_request_ids_are_never_reported_and_never_fail(fake: FakeEndor, request_id: str | None) -> None:
    def odd(request: httpx.Request) -> httpx.Response:
        r = fake.handler(request)
        headers = [(k, v) for k, v in r.headers.raw if k.lower() != b"x-request-id"]
        if request_id is not None:
            headers.append((b"x-request-id", request_id.encode("latin-1")))
        return httpx.Response(r.status_code, content=r.content, headers=headers)

    c = make_client(fake, odd)
    c.whoami()
    c.whoami()
    assert reported(fake) == [None, None]


def test_a_connection_error_reports_nothing(fake: FakeEndor) -> None:
    c = make_client(fake, retry=RetryPolicy(max_retries=0))
    c.whoami()
    fake.fail_next.append(httpx.ConnectError("reset"))
    with pytest.raises(Exception):  # noqa: B017
        c.whoami()  # took the first call's report with it
    c.whoami()
    assert reported(fake)[1] is not None and reported(fake)[2] is None
