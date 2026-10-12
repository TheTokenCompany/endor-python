"""User reports to the Endor team (``POST /v1/user_reports``): what didn't work as expected, what was unclear, and
what works.

    endor.user_report("system_one answered 422 for a choice with 2 options; the docs say 2 is fine", kind="bug")
    client.user_report("the integration runs: 120 decisions an hour", kind="works")

Meant for people and for their coding agents: every error's ``str()`` ends with a line saying how to report it
(``errors.report_hint``). A report works without an API key (it is then anonymous); with one, the Endor team sees
your org. It carries your message, the request it is about (by default the latest failed request of the client, or of
the process for ``endor.user_report``) and a small context: the SDK and Python versions, the platform, and the latest
failure's status, code and endpoint. Never your API key, a request body or a decision's content.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from typing import Any

import httpx

from . import _constants as C
from ._headers import _interface
from ._http import LAST, LastCall, Transport
from ._retry import RetryPolicy
from ._version import __version__
from .errors import ResponseValidationError
from .types import UserReportKind, UserReportReceipt

__all__ = ["user_report", "user_report_async", "KINDS"]

KINDS: tuple[str, ...] = ("bug", "confusing", "works")
MESSAGE_MAX = 4000
CONTEXT_MAX_KEYS = 30
CONTEXT_KEY_MAX = 64
CONTEXT_VALUE_MAX = 500
ENDPOINT = "/v1/user_reports"
# One retry for a lost connection or a 5xx; a 429 means enough reports were sent, so it is never retried.
RETRY = RetryPolicy(max_retries=1, http_statuses=frozenset({408, 500, 502, 503, 504}))


def body(
    message: str,
    kind: str,
    request_id: str | None,
    context: Mapping[str, Any] | None,
    last: LastCall,
) -> dict[str, Any]:
    """The request body: the message, the request it is about and the context (the module docstring)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)} (got {kind!r})")
    text = (message or "").strip()
    if not text:
        raise ValueError("message is empty: say what you did, what happened and what you expected")
    if len(text) > MESSAGE_MAX:
        text = text[: MESSAGE_MAX - 1] + "…"
    trouble = kind != "works"  # a "works" report is about the integration, not the latest failure
    if request_id is None:
        request_id = (last.failed_request_id or last.request_id) if trouble else last.request_id
    ctx: dict[str, Any] = {
        "sdk": C.SDK_NAME,
        "sdk_version": __version__,
        "python": platform.python_version(),
        "platform": f"{sys.platform}; {platform.machine() or 'unknown'}",
        "interface": _interface.get(),
    }
    if trouble and last.error:
        ctx.update({f"last_error_{k}": v for k, v in last.error.items() if v is not None})
    for k, v in (context or {}).items():
        ctx[str(k)[:CONTEXT_KEY_MAX]] = v
    return {"kind": kind, "message": text, "request_id": request_id, "context": _clean(ctx)}


def _clean(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """At most CONTEXT_MAX_KEYS keys; values as strings, numbers, booleans or None, strings cut at
    CONTEXT_VALUE_MAX."""
    out: dict[str, Any] = {}
    for k, v in list(ctx.items())[:CONTEXT_MAX_KEYS]:
        out[k] = v if v is None or isinstance(v, (bool, int, float)) else str(v)[:CONTEXT_VALUE_MAX]
    return out


def _last(t: Transport) -> LastCall:
    """The transport's latest requests, or the process's when it has made none (a report from a fresh client)."""
    return t.last if t.last.request_id or t.last.error else LAST


def _receipt(raw: Any) -> UserReportReceipt:
    try:
        return UserReportReceipt.model_validate(raw)
    except ValueError as e:
        raise ResponseValidationError(201, raw, None, str(e).splitlines()[0], f"POST {ENDPOINT}") from e


def send(
    t: Transport,
    message: str,
    kind: str,
    request_id: str | None,
    context: Mapping[str, Any] | None,
    method_name: str,
) -> UserReportReceipt:
    """Send a report through a transport (a client's, or one made for ``report``)."""
    raw = t.request(
        "POST",
        ENDPOINT,
        json=body(message, kind, request_id, context, _last(t)),
        method_name=method_name,
        idempotent=True,
        retry=RETRY,
        remember=False,
    )
    return _receipt(raw)


async def send_async(
    t: Transport,
    message: str,
    kind: str,
    request_id: str | None,
    context: Mapping[str, Any] | None,
    method_name: str,
) -> UserReportReceipt:
    raw = await t.arequest(
        "POST",
        ENDPOINT,
        json=body(message, kind, request_id, context, _last(t)),
        method_name=method_name,
        idempotent=True,
        retry=RETRY,
        remember=False,
    )
    return _receipt(raw)


def _transport(
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
    transport: httpx.BaseTransport | None,
    async_transport: httpx.AsyncBaseTransport | None,
) -> Transport:
    """A transport with the given key, else ENDOR_API_KEY, else none (anonymous): a report never fails for want of
    a key."""
    key = (api_key if api_key is not None else os.environ.get(C.API_KEY_ENV, "")).strip()
    if not key.isascii() or not key.isprintable() or " " in key:
        key = ""
    url = (base_url or os.environ.get(C.BASE_URL_ENV, "").strip() or C.DEFAULT_BASE_URL).rstrip("/")
    return Transport(
        key,
        url,
        timeout=C.DEFAULT_TIMEOUT if timeout is None else float(timeout),
        transport=transport,
        async_transport=async_transport,
    )


def user_report(
    message: str,
    *,
    kind: UserReportKind = "bug",
    request_id: str | None = None,
    context: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    transport: httpx.BaseTransport | None = None,
) -> UserReportReceipt:
    """Tell the Endor team that something didn't work as expected, was unclear, or works. No client needed, and no
    API key: with ``api_key`` (or ``ENDOR_API_KEY``) the team sees your org, without one the report is anonymous.

    Args:
        message: What you did, what happened and what you expected (up to 4000 characters; longer is cut).
        kind: ``"bug"`` (Endor did something unexpected), ``"confusing"`` (the docs, a message or the API are
            unclear) or ``"works"`` (the integration runs).
        request_id: The ``req_...`` id of the request it is about. Default: the latest failed request of any client
            in this process (for ``"works"``, the latest request).
        context: Extra details as short values (versions, settings, counts): at most 30 keys in all, strings cut at
            500 characters. Never put secrets or customer data here.
        api_key, base_url: Default to ``ENDOR_API_KEY`` (none: anonymous) and ``ENDOR_BASE_URL``, then production.
        timeout: Seconds for the request (default 30).
        transport: A custom ``httpx`` transport (mostly for tests).

    Returns:
        A ``UserReportReceipt`` with the report's ``id``.

    Raises:
        ValueError: An empty message, or an unknown ``kind``.
        RateLimitError: Too many reports from this org or address for now (``retry_after``); never retried.
    """
    t = _transport(api_key, base_url, timeout, transport, None)
    try:
        return send(t, message, kind, request_id, context, "endor.user_report")
    finally:
        t.close()


async def user_report_async(
    message: str,
    *,
    kind: UserReportKind = "bug",
    request_id: str | None = None,
    context: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    async_transport: httpx.AsyncBaseTransport | None = None,
) -> UserReportReceipt:
    """``report`` for async code."""
    t = _transport(api_key, base_url, timeout, None, async_transport)
    try:
        return await send_async(t, message, kind, request_id, context, "endor.user_report")
    finally:
        await t.aclose()
