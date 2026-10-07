"""Retries, timeouts, gzip and the mapping of error responses to exceptions."""

from __future__ import annotations

import gzip
import json
import time

import httpx
import pytest

import endor
from endor import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    EndorClient,
    InsufficientBalanceError,
    InternalServerError,
    NotFoundError,
    OverloadedError,
    PayloadTooLargeError,
    PermissionDeniedError,
    RateLimitError,
    RetryPolicy,
    UnprocessableEntityError,
)
from endor._retry import retry_after_seconds

from .conftest import API_KEY, BASE_URL, rows
from .fake_api import FakeEndor, HTTPError


def make_client(fake: FakeEndor, **kw: object) -> EndorClient:
    kw.setdefault("retry", RetryPolicy(backoff_initial=0.0, backoff_max=0.0))
    return EndorClient(api_key=API_KEY, base_url=BASE_URL, transport=httpx.MockTransport(fake.handler), **kw)  # type: ignore[arg-type]


class TestRetries:
    def test_retries_5xx_and_connection_errors(self, client: EndorClient, fake: FakeEndor) -> None:
        fake.fail_next += [HTTPError(500, "internal", "bug"), httpx.ConnectError("reset")]
        assert client.whoami().org_id == "org_test"
        assert len(fake.requests) == 3

    def test_retry_after_header_is_honored(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(backoff_initial=5.0, backoff_max=5.0))
        fake.fail_next.append(HTTPError(429, "rate_limited", "slow down", headers={"retry-after-ms": "20"}))
        t0 = time.monotonic()
        c.whoami()
        elapsed = time.monotonic() - t0
        assert 0.015 < elapsed < 1.0, elapsed

    def test_gives_up_after_max_retries(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=1, backoff_initial=0, backoff_max=0))
        fake.fail_next += [HTTPError(503, "unavailable", "a"), HTTPError(503, "unavailable", "b")]
        with pytest.raises(InternalServerError) as e:
            c.whoami()
        assert e.value.status == 503 and e.value.message == "b" and len(fake.requests) == 2

    def test_no_retries(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=0))
        fake.fail_next.append(HTTPError(500, "internal", "bug"))
        with pytest.raises(InternalServerError):
            c.whoami()
        assert len(fake.requests) == 1

    def test_4xx_is_not_retried(self, client: EndorClient, fake: FakeEndor) -> None:
        fake.fail_next.append(HTTPError(409, "conflict", "taken"))
        with pytest.raises(ConflictError):
            client.whoami()
        assert len(fake.requests) == 1

    def test_connection_error_surfaces_after_retries(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=1, backoff_initial=0, backoff_max=0))
        fake.fail_next += [httpx.ConnectError("a"), httpx.ConnectError("b")]
        with pytest.raises(APIConnectionError, match="ConnectError"):
            c.whoami()

    def test_timeout_maps_to_timeout_error(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=0))
        fake.fail_next.append(httpx.ReadTimeout("slow"))
        with pytest.raises(APITimeoutError) as e:
            c.whoami()
        assert isinstance(e.value, TimeoutError) and "timeout=30.0" in str(e.value)

    def test_timeout_retry_can_be_disabled(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(api_timeout_error=False, backoff_initial=0, backoff_max=0))
        fake.fail_next.append(httpx.ReadTimeout("slow"))
        with pytest.raises(APITimeoutError):
            c.whoami()
        assert len(fake.requests) == 1

    def test_total_budget_stops_retries(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=5, backoff_initial=10, backoff_max=10, timeout=1.0))
        fake.fail_next += [HTTPError(503, "unavailable", "a")] * 6
        t0 = time.monotonic()
        with pytest.raises(InternalServerError):
            c.whoami()
        assert time.monotonic() - t0 < 0.5 and len(fake.requests) == 1

    def test_per_call_retry_override(self, client: EndorClient, fake: FakeEndor) -> None:
        fake.fail_next.append(HTTPError(500, "internal", "bug"))
        with pytest.raises(InternalServerError):
            client.models.list(retry=RetryPolicy(max_retries=0))

    async def test_async_retries(self, client: EndorClient, fake: FakeEndor) -> None:
        fake.fail_next += [HTTPError(502, "bad_gateway", "x"), httpx.ConnectError("y")]
        res = await client.system_one_async("x", {"u": endor.Noul()}, model="jev-9b")
        assert res.nouls["u"].noul == 0.5 and len(fake.requests) == 3


class TestRetryPolicy:
    def test_validation(self) -> None:
        for bad in (dict(max_retries=-1), dict(backoff_initial=-1), dict(backoff_jitter=2), dict(timeout=0)):
            with pytest.raises(ValueError):
                RetryPolicy(**bad)  # type: ignore[arg-type]

    def test_backoff_grows_and_caps(self) -> None:
        p = RetryPolicy(backoff_initial=1, backoff_max=4, backoff_jitter=0)
        assert [p.backoff(i) for i in (1, 2, 3, 4)] == [1, 2, 4, 4]
        assert RetryPolicy(backoff_initial=0).backoff(3) == 0.0

    def test_retry_after_parsing(self) -> None:
        assert retry_after_seconds(httpx.Headers({"retry-after": "2"})) == 2.0
        assert retry_after_seconds(httpx.Headers({"retry-after-ms": "250"})) == 0.25
        assert retry_after_seconds(httpx.Headers({"retry-after": "-1"})) is None
        assert retry_after_seconds(httpx.Headers({"retry-after": "garbage"})) is None
        http_date = retry_after_seconds(httpx.Headers({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}))
        assert http_date == 0.0
        assert retry_after_seconds(httpx.Headers()) is None


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "cls"),
        [
            (400, BadRequestError),
            (401, AuthenticationError),
            (402, InsufficientBalanceError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (409, ConflictError),
            (413, PayloadTooLargeError),
            (422, UnprocessableEntityError),
            (429, RateLimitError),
            (529, OverloadedError),
            (500, InternalServerError),
            (418, APIError),
        ],
    )
    def test_status_mapping(self, fake: FakeEndor, status: int, cls: type[APIError]) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=0))
        fake.fail_next.append(HTTPError(status, "some_code", "what happened", param="field.x"))
        with pytest.raises(cls) as e:
            c.whoami()
        err = e.value
        assert err.status == status and err.code == "some_code" and err.param == "field.x"
        assert err.message == "what happened" and err.endpoint == "GET /v1/whoami"
        assert err.request_id and err.request_id.startswith("req_")
        assert (
            str(err)
            == f"GET /v1/whoami: {status} some_code: what happened (param=field.x) (request_id={err.request_id})"
        )
        assert repr(err).startswith(cls.__name__)
        assert isinstance(err, endor.EndorError)

    def test_rate_limit_retry_after(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=0))
        fake.fail_next.append(HTTPError(429, "quota_exceeded", "wait", headers={"retry-after": "7"}))
        with pytest.raises(RateLimitError) as e:
            c.whoami()
        assert e.value.retry_after == 7.0 and e.value.code == "quota_exceeded"

    def test_quota_and_balance_errors_are_not_retried(self, fake: FakeEndor) -> None:
        c = make_client(fake, retry=RetryPolicy(max_retries=3, backoff_initial=0, backoff_max=0, timeout=None))
        fake.fail_next.append(HTTPError(429, "quota_exceeded", "close a run", headers={"retry-after": "0"}))
        with pytest.raises(RateLimitError):
            c.whoami()
        fake.fail_next.append(HTTPError(402, "insufficient_balance", "add credit"))
        with pytest.raises(InsufficientBalanceError) as e:
            c.whoami()
        assert e.value.status == 402 and len(fake.requests) == 2
        fake.fail_next.append(HTTPError(429, "rate_limited", "slow down", headers={"retry-after": "0"}))
        c.whoami()  # a plain rate limit is retried
        assert len(fake.requests) == 4

    def test_balance_error_is_mapped_from_its_code(self) -> None:
        err = endor.errors.api_error(400, {"error": {"code": "insufficient_balance", "message": "m"}}, httpx.Headers())
        assert isinstance(err, InsufficientBalanceError)

    def test_bad_key_is_authentication_error(self, fake: FakeEndor) -> None:
        c = make_client(fake)
        c.api_key = c._t.api_key = "edk_wrong"
        with pytest.raises(AuthenticationError) as e:
            c.whoami()
        assert e.value.code == "unauthorized"

    def test_non_json_and_empty_bodies(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("text"):
                return httpx.Response(502, text="<html>bad gateway</html>")
            return httpx.Response(500)

        c = EndorClient(
            api_key="edk_x", base_url=BASE_URL, transport=httpx.MockTransport(handler), retry=RetryPolicy(max_retries=0)
        )
        with pytest.raises(InternalServerError, match="bad gateway"):
            c._t.request("GET", "/v1/text")
        with pytest.raises(InternalServerError, match="no response body"):
            c._t.request("GET", "/v1/empty")

    def test_unexpected_error_shapes(self) -> None:
        err = APIError(500, {"message": "flat"}, httpx.Headers())
        assert err.message == "flat" and err.code is None and str(err) == "500: flat"
        err = APIError(500, {"error": "plain string"}, None)
        assert err.message == "plain string"
        err = APIError(500, {"weird": "x" * 300}, None)
        assert err.message.endswith("…") and len(err.message) <= 201


class TestEncoding:
    def test_gzip_on_by_default_for_large_bodies(
        self, client: EndorClient, project: endor.Project, fake: FakeEndor
    ) -> None:
        project.datasets.upload("small", rows(2))
        small = fake.requests[-1]
        assert "content-encoding" not in small.headers and small.headers["content-type"] == "application/json"
        project.datasets.upload("big", rows(300))
        big = fake.requests[-1]
        assert big.headers["content-encoding"] == "gzip" and big.body["name"] == "big"
        assert fake.datasets[(project.name, "big")]["n_rows"] == 300

    def test_gzip_bytes_on_the_wire(self, fake: FakeEndor) -> None:
        seen: list[tuple[dict[str, str], bytes]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((dict(request.headers), request.content))
            return fake.handler(request)

        c = EndorClient(api_key=API_KEY, base_url=BASE_URL, transport=httpx.MockTransport(handler))
        p = c.projects.create("gz")
        p.datasets.upload("d", rows(300))
        headers, raw = seen[-1]
        assert headers["content-encoding"] == "gzip"
        assert json.loads(gzip.decompress(raw))["name"] == "d" and len(raw) < 20_000

    def test_gzip_opt_out(self, fake: FakeEndor) -> None:
        c = EndorClient(api_key=API_KEY, base_url=BASE_URL, transport=httpx.MockTransport(fake.handler), gzip=False)
        p = c.projects.create("plain")
        p.datasets.upload("d", rows(300))
        assert "content-encoding" not in fake.requests[-1].headers

    def test_none_params_are_dropped(self, client: EndorClient, fake: FakeEndor, project: endor.Project) -> None:
        project.runs.list()
        assert fake.requests[-1].params == {"limit": "100", "offset": "0"}
        project.runs.list(tag="x", limit=5)
        assert fake.requests[-1].params == {"limit": "5", "offset": "0", "tag": "x"}

    def test_lists_page_through_everything(
        self, client: EndorClient, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(endor._constants, "LIST_PAGE_SIZE", 2)
        names = {client.projects.create(f"page-{i}").name for i in range(5)}
        assert {p.name for p in client.projects.list()} == names
        pages = [r.params for r in fake.requests if r.method == "GET" and r.path == "/v1/projects"]
        assert [p["offset"] for p in pages] == ["0", "2", "4"]
        assert len(client.projects.list(limit=3)) == 3
