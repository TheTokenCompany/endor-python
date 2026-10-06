"""Exceptions.

    EndorError
      APIError (status, code, param, message, body, headers, endpoint, request_id)
        BadRequestError 400 · AuthenticationError 401 · PermissionDeniedError 403 · NotFoundError 404
        ConflictError 409 · PayloadTooLargeError 413 · UnprocessableEntityError 422
        RateLimitError 429 (retry_after) · OverloadedError 529 · InternalServerError 5xx
        ResponseValidationError: a 2xx whose body does not have the expected shape
      APIConnectionError: no HTTP response
        APITimeoutError
      OperationFailedError (code, future): a training or evaluation operation failed or was cancelled

Every API error body is ``{"error": {"code": "<stable_code>", "message": "<for people>", "param"?: "<field>"}}``.
``APIError.code`` is the stable code, meant for programs; ``str(error)`` is for people.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ._constants import HEADER_REQUEST_ID, MAX_ERROR_BODY_LENGTH
from ._retry import retry_after_seconds

__all__ = [
    "EndorError",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "PayloadTooLargeError",
    "UnprocessableEntityError",
    "RateLimitError",
    "OverloadedError",
    "InternalServerError",
    "ResponseValidationError",
    "APIConnectionError",
    "APITimeoutError",
    "OperationFailedError",
    "api_error",
]


class EndorError(Exception):
    """Base class for every error the SDK raises."""


class APIError(EndorError):
    """An unsuccessful HTTP response."""

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx.Headers | None = None,
        message: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.status = status
        """HTTP status code."""
        self.body = body
        """The decoded JSON error body, the response text, or None."""
        self.headers = headers if headers is not None else httpx.Headers()
        """Response headers."""
        self.endpoint = endpoint
        """``"<METHOD> <path>"`` of the failed request, without credentials or query string."""
        err: dict[str, Any] = {}
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
        code, param = err.get("code"), err.get("param")
        self.code: str | None = code if isinstance(code, str) else None
        """Endor's stable error code (``not_found``, ``invalid_datum``, ``seq_conflict`` ...), or None."""
        self.param: str | None = param if isinstance(param, str) else None
        """The request field the error is about (``data[3].target``), when the server names one."""
        self.message = message or _extract_message(body) or ("no response body" if body is None else _truncate(body))
        super().__init__(self.message)

    @property
    def request_id(self) -> str | None:
        """The server's ``x-request-id`` for this response, or None."""
        v = self.headers.get(HEADER_REQUEST_ID)
        return str(v) if v is not None else None

    def __str__(self) -> str:
        parts = [str(self.status)]
        if self.code:
            parts.append(self.code)
        text = " ".join(parts) + f": {self.message}"
        if self.param:
            text += f" (param={self.param})"
        if self.endpoint:
            text = f"{self.endpoint}: {text}"
        if self.request_id:
            text += f" (request_id={self.request_id})"
        return text

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class BadRequestError(APIError):
    """400: the request was invalid."""


class AuthenticationError(APIError):
    """401: missing, unknown or revoked API key."""


class PermissionDeniedError(APIError):
    """403: access denied."""


class NotFoundError(APIError):
    """404: the resource does not exist, or is not yours."""


class ConflictError(APIError):
    """409: a name is taken, a run is closed, or a sequence number is out of order (see ``code``)."""


class PayloadTooLargeError(APIError):
    """413: the request body is over the server's limit."""


class UnprocessableEntityError(APIError):
    """422: validation failed; ``param`` or the message names the field or item."""


class RateLimitError(APIError):
    """429: rate limited, queue full or quota exceeded (see ``code``). Honors ``Retry-After``."""

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx.Headers | None = None,
        message: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(status, body, headers, message, endpoint)
        self.retry_after: float | None = retry_after_seconds(self.headers)
        """Seconds the server asked us to wait, or None."""


class OverloadedError(APIError):
    """529: decision servers are saturated; retry later."""


class InternalServerError(APIError):
    """5xx: the server failed."""


class ResponseValidationError(APIError):
    """A successful response whose body did not have the expected shape."""

    def __init__(self, status: int, body: Any, headers: httpx.Headers | None, detail: str, endpoint: str | None):
        super().__init__(status, body, headers, f"invalid response data: {detail}", endpoint)


class APIConnectionError(EndorError, ConnectionError):
    """The request got no HTTP response."""


class APITimeoutError(APIConnectionError, TimeoutError):
    """The request exceeded its timeout."""

    def __init__(self, timeout: float | httpx.Timeout | None) -> None:
        super().__init__(f"request timed out (timeout={timeout})")
        self.timeout = timeout


_HINTS = {
    "trainer_lost": "The run's trainer is gone. Start a new run from your last saved model: "
    "project.runs.create(from_model=...).",
    "oom": "The trainer ran out of memory. Retry with a smaller batch.",
    "no_gradients": "optim_step found no accumulated gradients. Call forward_backward first, with non-zero weights.",
}


class OperationFailedError(EndorError):
    """A training or evaluation operation (a future) finished ``failed`` or ``cancelled``."""

    def __init__(self, message: str, code: str | None = None, future: dict[str, Any] | None = None) -> None:
        self.code = code
        """The server's stable code: ``trainer_lost``, ``oom``, ``no_gradients``, ``cancelled``, ``invalid_datum``,
        ``internal``."""
        self.future = future
        """The future's final state as returned by the server."""
        hint = _HINTS.get(code or "")
        self.message = f"{message} {hint}" if hint else message
        super().__init__(self.message)

    @property
    def retryable(self) -> bool:
        """Whether retrying the same operation on the same run makes sense (only ``oom`` with a smaller batch)."""
        return self.code == "oom"

    def __str__(self) -> str:
        return f"{self.code}: {self.message}" if self.code else self.message


_STATUS_ERRORS: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    413: PayloadTooLargeError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    529: OverloadedError,
}


def api_error(status: int, body: Any, headers: httpx.Headers, endpoint: str | None = None) -> APIError:
    """The exception for an HTTP error response."""
    cls = _STATUS_ERRORS.get(status, InternalServerError if status >= 500 else APIError)
    return cls(status, body, headers, endpoint=endpoint)


def _extract_message(body: Any) -> str | None:
    if isinstance(body, str):
        return body or None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, str):
        return err
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return str(err["message"])
    for key in ("message", "detail"):
        v = body.get(key)
        if isinstance(v, str):
            return v
    return None


def _truncate(body: Any) -> str:
    raw = body if isinstance(body, str) else json.dumps(body, default=str)
    return raw[:MAX_ERROR_BODY_LENGTH] + "…" if len(raw) > MAX_ERROR_BODY_LENGTH else raw
