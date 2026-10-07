"""RetryPolicy: how the SDK retries failed HTTP requests."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import httpx

from ._constants import HEADER_RETRY_AFTER, HEADER_RETRY_AFTER_MS


@dataclass(frozen=True)
class RetryPolicy:
    """Retry behavior for every SDK call.

    Examples:
        ```python
        from endor import EndorClient, RetryPolicy

        client = EndorClient(retry=RetryPolicy(max_retries=3, timeout=60.0))
        client = EndorClient(retry=RetryPolicy(max_retries=0))   # no retries
        ```
    """

    max_retries: int = 2
    """Retries after the first attempt; ``0`` disables retries."""

    backoff_initial: float = 0.5
    """First backoff delay in seconds, doubled each attempt up to ``backoff_max``."""

    backoff_max: float = 5.0
    """Longest backoff delay in seconds."""

    backoff_jitter: float = 0.25
    """Fraction of each delay randomly subtracted, between 0 and 1."""

    http_statuses: frozenset[int] = field(default_factory=lambda: frozenset({408, 429, *range(500, 600)}))
    """HTTP status codes that are retried. 4xx conflicts and validation errors are never retried."""

    respect_retry_after: bool = True
    """Honor ``Retry-After`` and ``retry-after-ms`` response headers."""

    api_connection_error: bool = True
    """Retry when the request cannot reach or read from the server."""

    api_timeout_error: bool = True
    """Retry when a request exceeds its timeout."""

    timeout: float | None = 60.0
    """Total budget in seconds for one SDK call, retries and delays included; ``None`` for no limit."""

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        for name in ("backoff_initial", "backoff_max"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a non-negative, finite number of seconds")
        if not 0 <= self.backoff_jitter <= 1:
            raise ValueError("backoff_jitter must be between 0 and 1")
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be a positive number of seconds, or None")

    # ------------------------------------------------------------------ decisions
    def retryable_status(self, status: int) -> bool:
        """Whether a response with this status code should be retried."""
        return status in self.http_statuses

    def retryable_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, httpx.TimeoutException):
            return self.api_timeout_error
        return isinstance(exc, httpx.TransportError) and self.api_connection_error

    def backoff(self, attempt: int) -> float:
        """Delay before retry number ``attempt`` (1-based), with jitter."""
        if self.backoff_initial == 0 or self.backoff_max == 0:
            return 0.0
        exponential = min(self.backoff_max, self.backoff_initial * 2 ** (attempt - 1))
        return float(round(exponential * (1 - random.random() * self.backoff_jitter), 3))

    def delay(self, attempt: int, headers: httpx.Headers | None) -> float:
        """The wait before retry ``attempt``: ``Retry-After`` when present and honored, else backoff."""
        if headers is not None and self.respect_retry_after:
            after = retry_after_seconds(headers)
            if after is not None:
                return after
        return self.backoff(attempt)

    def allows(self, attempt: int, started: float, delay: float) -> bool:
        """Whether retry ``attempt`` (1-based) fits the retry count and the total time budget."""
        if attempt > self.max_retries:
            return False
        if self.timeout is None:
            return True
        return time.monotonic() - started + delay < self.timeout


def retry_after_seconds(headers: httpx.Headers) -> float | None:
    """``retry-after-ms`` or ``Retry-After`` (seconds or an HTTP date) as seconds, or None."""
    raw = headers.get(HEADER_RETRY_AFTER_MS)
    if raw is not None:
        try:
            ms = float(raw.strip())
            if math.isfinite(ms) and ms >= 0:
                return ms / 1000
        except ValueError:
            pass
    raw = headers.get(HEADER_RETRY_AFTER)
    if raw is None:
        return None
    try:
        s = float(raw.strip())
        return s if math.isfinite(s) and s >= 0 else None
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None
