"""Exceptions.

    EndorError
      APIError (status, code, param, message, body, headers, endpoint, request_id)
        BadRequestError 400 · AuthenticationError 401 · InsufficientBalanceError 402 · PermissionDeniedError 403
        NotFoundError 404 (NoPreviewError, VersionNotCallableError)
        ConflictError 409 (LimitReachedError > ProductionFullError, ModelNotInProductionError,
            ModelInProductionError)
        PayloadTooLargeError 413
        UnprocessableEntityError 422 (ModelRequiresProjectError)
        RateLimitError 429 (retry_after) · OverloadedError 529 · InternalServerError 5xx
        ResponseValidationError: a 2xx whose body does not have the expected shape
      APIConnectionError: no HTTP response
        APITimeoutError
      OperationFailedError (code, future): a training or evaluation operation failed or was cancelled

Every API error body is ``{"error": {"code": "<stable_code>", "message": "<for people>", "param"?: "<field>"}}``.
``APIError.code`` is the stable code, meant for programs; ``str(error)`` is for people (and coding agents): it ends
with one line saying how to tell the Endor team when something didn't work as expected (``endor.user_report``).
``APIError.message`` is the server's message alone, for showing to your own users.
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
    "InsufficientBalanceError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "LimitReachedError",
    "ProductionFullError",
    "ModelNotInProductionError",
    "ModelInProductionError",
    "VersionNotCallableError",
    "NoPreviewError",
    "PayloadTooLargeError",
    "UnprocessableEntityError",
    "ModelRequiresProjectError",
    "RateLimitError",
    "OverloadedError",
    "InternalServerError",
    "ResponseValidationError",
    "APIConnectionError",
    "APITimeoutError",
    "OperationFailedError",
    "DownloadError",
    "api_error",
    "report_hint",
]

USER_REPORTS_ENDPOINT = "POST /v1/user_reports"


def report_hint(request_id: str | None = None, *, cli: bool = False) -> str:
    """The line that ends every error's ``str()``: how to tell the Endor team (``endor.user_report``, or
    ``endor user-report`` in the CLI), with the request id when there is one."""
    lead = "If Endor did something you didn't expect, or this message is unclear, tell the Endor team: "
    if cli:
        rid = f" --request-id {request_id}" if request_id else ""
        return lead + f'endor user-report "what you did, what happened, what you expected" --kind bug{rid}'
    rid = f', request_id="{request_id}"' if request_id else ""
    return lead + f'endor.user_report("what you did, what happened, what you expected", kind="bug"{rid})'


class EndorError(Exception):
    """Base class for every error the SDK raises. ``str(error)`` ends with ``report_hint``: one line on how to report
    a problem to the Endor team."""

    def _text(self) -> str:
        """The error itself, without the report line."""
        return super().__str__()

    def _hint_request_id(self) -> str | None:
        return None

    def _with_hint(self) -> bool:
        return True

    def __str__(self) -> str:
        text = self._text()
        return f"{text}\n{report_hint(self._hint_request_id())}" if self._with_hint() else text


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

    def _text(self) -> str:
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

    def _hint_request_id(self) -> str | None:
        return self.request_id

    def _with_hint(self) -> bool:
        return self.endpoint != USER_REPORTS_ENDPOINT  # a failed report says nothing about reporting it

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._text()!r})"


class BadRequestError(APIError):
    """400: the request was invalid."""


class AuthenticationError(APIError):
    """401: missing, unknown or revoked API key."""


class InsufficientBalanceError(APIError):
    """402 ``insufficient_balance``: the org's balance is used up (a new org starts with none). Paid calls (decisions,
    new runs, training calls, evaluations) are refused until you add credit in the dashboard under Billing. Never
    retried."""


class PermissionDeniedError(APIError):
    """403: access denied."""


class NotFoundError(APIError):
    """404: the resource does not exist, or is not yours."""


class NoPreviewError(NotFoundError):
    """404 ``no_preview``: ``"<project>:preview"`` was called, but the project has no preview. A project has one only
    while its newest version isn't in production, so never while auto-promote is on (each new version goes to
    production at once): call
    ``"<project>"``."""


class VersionNotCallableError(NotFoundError):
    """404 ``version_not_callable``: ``"<project>:v3"`` names an older version of a continuous learning project. Only
    ``"<project>"`` (the production version) and ``"<project>:preview"`` (a newer one) can be called; promote the
    version
    (``project.promote("v3")``) to call it as ``"<project>"``."""


class ConflictError(APIError):
    """409: a name is taken, a run is closed, a sequence number is out of order, or a limit is reached (see
    ``code``)."""


class LimitReachedError(ConflictError):
    """409 ``limit_reached``: a count limit of the org is reached, for example its projects, custom models, datasets or
    open runs. The
    message names the limit and its value (``client.whoami().limits`` has them all). Free one first: waiting doesn't
    help, so it is never retried."""


class ProductionFullError(LimitReachedError):
    """409 ``production_full``: every production slot for custom models is taken (``max_production_models_per_org`` in
    ``client.whoami().limits``). Promote with ``replace=`` naming the model to move to the archive in the same step, or
    archive one first. The message names the models in production."""


class ModelNotInProductionError(ConflictError):
    """409 ``model_not_in_production``: a decision named an archived custom model. Promote it first
    (``client.custom_models.promote(name)``); archived models can still be downloaded, evaluated and trained from."""


class ModelInProductionError(ConflictError):
    """409 ``model_in_production``: the call doesn't apply to a custom model in production (deleting it, or giving it
    a TTL: models in production never expire). Archive it first (``client.custom_models.archive(name)``)."""


class PayloadTooLargeError(APIError):
    """413: the request body is over the server's limit."""


class UnprocessableEntityError(APIError):
    """422: validation failed; ``param`` or the message names the field or item."""


class ModelRequiresProjectError(UnprocessableEntityError):
    """422 ``model_requires_project``: ``model`` is a bare base model id. Decisions name a continuous learning project
    (``"<project>"``, ``"<project>:preview"``) or a custom model in production by its name: create a project on the base
    (``client.projects.create(name, base_model=...)``) to decide with it."""


class RateLimitError(APIError):
    """429: rate limited or queue full (see ``code``). Honors ``Retry-After``."""

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
    "client.runs.create(from_model=...).",
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
        if hint and "new run" in message.lower():  # the server already says what to do
            hint = None
        self.message = f"{message} {hint}" if hint else message
        super().__init__(self.message)

    @property
    def retryable(self) -> bool:
        """Whether retrying the same operation on the same run makes sense (only ``oom`` with a smaller batch)."""
        return self.code == "oom"

    def _text(self) -> str:
        return f"{self.code}: {self.message}" if self.code else self.message


class DownloadError(EndorError):
    """A model file could not be downloaded, or failed its size or SHA-256 check. The message names the file, never
    the link."""


_STATUS_ERRORS: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: InsufficientBalanceError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    413: PayloadTooLargeError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    529: OverloadedError,
}


# Error codes with their own class, whatever the status.
_CODE_ERRORS: dict[str, type[APIError]] = {
    "insufficient_balance": InsufficientBalanceError,
    "limit_reached": LimitReachedError,
    "production_full": ProductionFullError,
    "model_not_in_production": ModelNotInProductionError,
    "model_in_production": ModelInProductionError,
    "version_not_callable": VersionNotCallableError,
    "no_preview": NoPreviewError,
    "model_requires_project": ModelRequiresProjectError,
}


def api_error(status: int, body: Any, headers: httpx.Headers, endpoint: str | None = None) -> APIError:
    """The exception for an HTTP error response."""
    err = body.get("error") if isinstance(body, dict) else None
    code = err.get("code") if isinstance(err, dict) else None
    cls = _CODE_ERRORS.get(code) if isinstance(code, str) else None
    if cls is None:
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
