"""HTTP transport: one place that sends every request, with retries, SDK headers and error mapping.

Each request carries the SDK identity headers (``_headers.py``), the public method that caused it
(``X-Endor-SDK-Method``), a client request id that stays the same across retries, and, for resource-creating
POSTs, an ``Idempotency-Key`` so a retried create never makes a duplicate.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic_core import to_json

from . import errors as E
from ._constants import (
    HEADER_CLIENT_REQUEST_ID,
    HEADER_IDEMPOTENCY_KEY,
    HEADER_REQUEST_ID,
    HEADER_RETRY_COUNT,
    NON_RETRYABLE_CODES,
)
from ._headers import context_headers, identity_headers
from ._log import logger, redact_headers
from ._retry import RetryPolicy

GZIP_THRESHOLD = 32 * 1024  # bytes; smaller bodies are not worth compressing


class Transport:
    """Sync and async HTTP for the Endor API. Not part of the public surface."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        retry: RetryPolicy | None = None,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
        gzip_bodies: bool = False,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryPolicy()
        self.timeout = timeout
        self.gzip_bodies = gzip_bodies
        self._user_headers = dict(headers or {})
        self._client = http_client
        self._aclient = async_http_client
        self._transport = transport
        self._atransport = async_transport
        self._owns_client = http_client is None
        self._owns_aclient = async_http_client is None

    # ------------------------------------------------------------------ clients
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self._transport)
        return self._client

    @property
    def aclient(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, transport=self._atransport)
        return self._aclient

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()

    async def aclose(self) -> None:
        if self._aclient is not None and self._owns_aclient:
            await self._aclient.aclose()

    # ------------------------------------------------------------------ request building
    def _headers(
        self,
        *,
        method_name: str | None,
        request_id: str,
        attempt: int,
        idempotency_key: str | None,
        extra: Mapping[str, str] | None,
        has_body: bool,
        compressed: bool,
    ) -> dict[str, str]:
        h: dict[str, str] = {**self._user_headers, **(extra or {})}
        h.update(identity_headers())
        h.update(context_headers(method_name))
        h["Authorization"] = f"Bearer {self.api_key}"
        h["Accept"] = "application/json"
        h[HEADER_CLIENT_REQUEST_ID] = request_id
        if attempt:
            h[HEADER_RETRY_COUNT] = str(attempt)
        else:
            h.pop(HEADER_RETRY_COUNT, None)
        if idempotency_key:
            h[HEADER_IDEMPOTENCY_KEY] = idempotency_key
        if has_body:
            h["Content-Type"] = "application/json"
            if compressed:
                h["Content-Encoding"] = "gzip"
        return h

    def _encode(self, body: Any) -> tuple[bytes | None, bool]:
        if body is None:
            return None, False
        raw = to_json(body)
        if self.gzip_bodies and len(raw) > GZIP_THRESHOLD:
            return gzip.compress(raw, compresslevel=6), True
        return raw, False

    @staticmethod
    def _url(base_url: str, path: str) -> str:
        return path if path.startswith("http") else base_url + path

    def _result(self, r: httpx.Response, endpoint: str) -> Any:
        if (r.status_code == 204 or not r.content) and r.is_success:
            return None
        try:
            body: Any = r.json()
        except ValueError:
            body = r.text or None
        if r.is_success:
            return body
        raise E.api_error(r.status_code, body, r.headers, endpoint)

    @staticmethod
    def _network_error(exc: Exception, timeout: float | httpx.Timeout | None) -> E.APIConnectionError:
        if isinstance(exc, httpx.TimeoutException):
            return E.APITimeoutError(timeout)
        return E.APIConnectionError(f"connection error: {type(exc).__name__}: {exc}")

    def _log(self, method: str, path: str, r: httpx.Response | None, started: float, exc: BaseException | None) -> None:
        ms = (time.monotonic() - started) * 1000
        if r is not None:
            logger.info(
                "%s %s <- %s in %.0fms (request %s)",
                method,
                path,
                r.status_code,
                ms,
                r.headers.get(HEADER_REQUEST_ID, "-"),
            )
        else:
            logger.info("%s %s <- %s in %.0fms", method, path, type(exc).__name__, ms)

    # ------------------------------------------------------------------ sync
    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        method_name: str | None = None,
        idempotent: bool = False,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Send one request with retries. Returns the decoded JSON body (None for 204); raises ``APIError``."""
        policy = retry or self.retry
        endpoint = f"{method} {path}"
        request_id = str(uuid.uuid4())
        idem = str(uuid.uuid4()) if idempotent else None
        content, compressed = self._encode(json)
        started, attempt = time.monotonic(), 0
        while True:
            headers = self._headers(
                method_name=method_name,
                request_id=request_id,
                attempt=attempt,
                idempotency_key=idem,
                extra=extra_headers,
                has_body=content is not None,
                compressed=compressed,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("%s %s -> headers=%s", method, path, redact_headers(headers))
            t0 = time.monotonic()
            try:
                r = self.client.request(
                    method,
                    self._url(self.base_url, path),
                    content=content,
                    params=_clean(params),
                    headers=headers,
                    timeout=timeout if timeout is not None else self.timeout,
                )
            except httpx.TransportError as exc:
                self._log(method, path, None, t0, exc)
                delay = policy.delay(attempt + 1, None)
                if not (policy.retryable_exception(exc) and policy.allows(attempt + 1, started, delay)):
                    raise self._network_error(exc, timeout if timeout is not None else self.timeout) from exc
                time.sleep(delay)
            else:
                self._log(method, path, r, t0, None)
                if r.is_success:
                    return self._result(r, endpoint)
                delay = policy.delay(attempt + 1, r.headers)
                if not (
                    policy.retryable_status(r.status_code)
                    and not _final_error(r)
                    and policy.allows(attempt + 1, started, delay)
                ):
                    return self._result(r, endpoint)
                time.sleep(delay)
            attempt += 1
            logger.info("%s %s retry %d", method, path, attempt)

    # ------------------------------------------------------------------ async
    async def arequest(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        method_name: str | None = None,
        idempotent: bool = False,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> Any:
        policy = retry or self.retry
        endpoint = f"{method} {path}"
        request_id = str(uuid.uuid4())
        idem = str(uuid.uuid4()) if idempotent else None
        content, compressed = self._encode(json)
        started, attempt = time.monotonic(), 0
        while True:
            headers = self._headers(
                method_name=method_name,
                request_id=request_id,
                attempt=attempt,
                idempotency_key=idem,
                extra=extra_headers,
                has_body=content is not None,
                compressed=compressed,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("%s %s -> headers=%s", method, path, redact_headers(headers))
            t0 = time.monotonic()
            try:
                r = await self.aclient.request(
                    method,
                    self._url(self.base_url, path),
                    content=content,
                    params=_clean(params),
                    headers=headers,
                    timeout=timeout if timeout is not None else self.timeout,
                )
            except httpx.TransportError as exc:
                self._log(method, path, None, t0, exc)
                delay = policy.delay(attempt + 1, None)
                if not (policy.retryable_exception(exc) and policy.allows(attempt + 1, started, delay)):
                    raise self._network_error(exc, timeout if timeout is not None else self.timeout) from exc
                await asyncio.sleep(delay)
            else:
                self._log(method, path, r, t0, None)
                if r.is_success:
                    return self._result(r, endpoint)
                delay = policy.delay(attempt + 1, r.headers)
                if not (
                    policy.retryable_status(r.status_code)
                    and not _final_error(r)
                    and policy.allows(attempt + 1, started, delay)
                ):
                    return self._result(r, endpoint)
                await asyncio.sleep(delay)
            attempt += 1
            logger.info("%s %s retry %d", method, path, attempt)


def _final_error(r: httpx.Response) -> bool:
    """Whether the error body carries a code that retrying can't fix (``limit_reached``, ``insufficient_balance``)."""
    try:
        err = r.json().get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(err, dict) and err.get("code") in NON_RETRYABLE_CODES


def _clean(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Drop None-valued query parameters."""
    if not params:
        return None
    out = {k: v for k, v in params.items() if v is not None}
    return out or None
