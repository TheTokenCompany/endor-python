"""A response with a ``Deprecation`` header (RFC 9745) raises a ``DeprecationWarning`` with the API's
``Endor-Warning`` text, once per distinct message per process, sync and async, pointing at the caller's line; a
response without one raises nothing."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator

import httpx
import pytest

from endor import EndorClient, RetryPolicy, _http

from .conftest import API_KEY, BASE_URL
from .fake_api import FakeEndor

OLD = 'model "tickets/v2" is now "tickets-v2"; the old name stops working on 2026-11-30'


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    _http._warned.clear()  # once per process: each test starts as a new process would
    yield
    _http._warned.clear()


def make_client(fake: FakeEndor, headers: dict[str, str]) -> EndorClient:
    def handler(request: httpx.Request) -> httpx.Response:
        r = fake.handler(request)
        r.headers.update(headers)
        return r

    return EndorClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
        retry=RetryPolicy(backoff_initial=0.0, backoff_max=0.0),
    )


def deprecations(call: Callable[[], object]) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        call()
    return [w for w in got if issubclass(w.category, DeprecationWarning)]


def test_a_deprecated_response_warns_once_per_message(fake: FakeEndor) -> None:
    c = make_client(
        fake, {"Deprecation": "@1792540800", "Sunset": "Mon, 30 Nov 2026 00:00:00 GMT", "Endor-Warning": OLD}
    )
    got = deprecations(lambda: c.whoami())
    assert [str(w.message) for w in got] == [OLD]
    assert got[0].filename == __file__  # the caller's line, not the SDK's
    assert deprecations(lambda: c.whoami()) == []  # the same message: once per process

    other = make_client(fake, {"Deprecation": "@1792540800", "Endor-Warning": 'model "a/b" is now "a-b"'})
    assert [str(w.message) for w in deprecations(lambda: other.whoami())] == ['model "a/b" is now "a-b"']


def test_without_endor_warning_it_names_the_endpoint_and_sunset(fake: FakeEndor) -> None:
    c = make_client(fake, {"Deprecation": "@1792540800", "Sunset": "Mon, 30 Nov 2026 00:00:00 GMT"})
    got = deprecations(lambda: c.whoami())
    assert [str(w.message) for w in got] == ["GET /v1/whoami is deprecated (sunset: Mon, 30 Nov 2026 00:00:00 GMT)"]


def test_no_deprecation_header_no_warning(fake: FakeEndor) -> None:
    c = make_client(fake, {"Endor-Warning": OLD})  # the text alone is not a deprecation
    assert deprecations(lambda: c.whoami()) == []


async def test_async_calls_warn_too(fake: FakeEndor) -> None:
    c = make_client(fake, {"Deprecation": "@1792540800", "Endor-Warning": OLD})
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        await c.models.list_async()
        await c.models.list_async()
    msgs = [w for w in got if issubclass(w.category, DeprecationWarning)]
    assert [str(w.message) for w in msgs] == [OLD] and msgs[0].filename == __file__
