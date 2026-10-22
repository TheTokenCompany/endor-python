"""APIFuture: the handle every slow operation returns.

Submitting is immediate; ``.result()`` (or ``await future``) waits. Submit several operations before waiting on any
of them (``forward_backward`` then ``optim_step``) and the trainer runs them back to back.

    fb = run.forward_backward(batch)
    opt = run.optim_step(learning_rate=1e-4)
    out, step = endor.gather(fb, opt)          # one long-poll for both

In a training loop, ``Steps`` sends the next step before the last one's results are read, so the trainer never waits
for your round trip between steps.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Generator
from typing import Any, Generic, TypeVar

from . import errors as E
from ._constants import FUTURE_POLL_WAIT_S, MAX_FUTURES_PER_RETRIEVE
from ._http import Transport
from ._parse import parse, validated
from .types import FutureInfo

__all__ = ["APIFuture", "Steps", "gather", "gather_async"]

T = TypeVar("T")
_UNSET: Any = object()


def _identity(x: Any) -> Any:
    return x


class APIFuture(Generic[T]):
    """The result of an operation the server runs for you. ``result()`` blocks; ``await`` works in async code.

    A future is either a leaf (one server future id) or a combination of children (a chunked batch). Both settle
    to a value or raise ``OperationFailedError`` with the server's stable code.
    """

    def __init__(
        self,
        transport: Transport | None,
        future_id: str | None = None,
        parse: Callable[[Any], T] = _identity,
        *,
        children: list[APIFuture[Any]] | None = None,
        combine: Callable[[list[Any]], T] | None = None,
        value: Any = _UNSET,
        method: str = "future.result",
    ) -> None:
        self._t = transport
        self.id = future_id
        """The server's future id (None for a combined or already-completed future)."""
        self._parse = parse
        self._children = children
        self._combine = combine
        self._value: Any = value
        self._exc: BaseException | None = None
        self._method = method
        self._info: FutureInfo | None = None

    @classmethod
    def completed(cls, value: T) -> APIFuture[T]:
        """A future that already holds ``value``."""
        return cls(None, value=value)

    def __repr__(self) -> str:
        return f"APIFuture(id={self.id!r}, done={self.done()})"

    # ------------------------------------------------------------------ state
    def done(self) -> bool:
        """Whether the result (or failure) is already known locally."""
        return self._value is not _UNSET or self._exc is not None

    @property
    def info(self) -> FutureInfo | None:
        """The server's last reported state of this future, if it has been polled."""
        return self._info

    def leaves(self) -> list[APIFuture[Any]]:
        """The unsettled server futures this future depends on."""
        if self.done():
            return []
        if self._children is not None:
            return [leaf for c in self._children for leaf in c.leaves()]
        return [self] if self.id else []

    def _settle(self, f: dict[str, Any], endpoint: str | None = None) -> bool:
        """Record a server state (from ``endpoint``); True once settled. A result without the expected shape settles
        it with ``ResponseValidationError``."""
        endpoint = endpoint or f"GET /v1/futures/{self.id}"
        self._info = parse(FutureInfo, f, endpoint)
        status = f.get("status")
        if status == "completed":
            try:
                self._value = validated(self._parse, f.get("result") or {}, endpoint)
            except E.ResponseValidationError as e:
                self._exc = e
            return True
        if status in ("failed", "cancelled"):
            err = f.get("error") or {}
            code = err.get("code") or ("cancelled" if status == "cancelled" else None)
            self._exc = E.OperationFailedError(err.get("message") or f"operation {status}", code=code, future=f)
            return True
        return False

    def _finish(self) -> T:
        if self._exc is not None:
            raise self._exc
        if self._value is _UNSET and self._children is not None:
            assert self._combine is not None
            self._value = self._combine([c._finish() for c in self._children])
        return self._value  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ waiting
    def result(self, timeout: float | None = None) -> T:
        """Block until the operation finishes. Raises ``OperationFailedError`` if it failed or was cancelled,
        ``TimeoutError`` when ``timeout`` seconds pass first (the operation keeps running)."""
        leaves = self.leaves()
        if leaves:
            assert self._t is not None
            _wait_all(self._t, leaves, timeout, self._method)
        return self._finish()

    async def result_async(self, timeout: float | None = None) -> T:
        """``result()`` for async code."""
        leaves = self.leaves()
        if leaves:
            assert self._t is not None
            await _wait_all_async(self._t, leaves, timeout, self._method)
        return self._finish()

    def __await__(self) -> Generator[Any, None, T]:
        return self.result_async().__await__()

    def cancel(self) -> None:
        """Cancel the operation if it has not started. A running operation finishes; a finished one is unchanged."""
        for leaf in self.leaves():
            assert leaf._t is not None
            leaf._t.request("POST", f"/v1/futures/{leaf.id}/cancel", method_name="future.cancel")

    async def cancel_async(self) -> None:
        """``cancel()`` for async code."""
        for leaf in self.leaves():
            assert leaf._t is not None
            await leaf._t.arequest("POST", f"/v1/futures/{leaf.id}/cancel", method_name="future.cancel")


# ---------------------------------------------------------------- polling


def _wait_s(deadline: float | None) -> float:
    """Server-side wait for the next poll: 25 s, or less when the caller's ``timeout`` runs out sooner. The HTTP
    request timeout is the client's timeout plus this wait."""
    if deadline is None:
        return FUTURE_POLL_WAIT_S
    return max(0.0, min(FUTURE_POLL_WAIT_S, deadline - time.monotonic()))


def _timeout_error(pending: list[APIFuture[Any]]) -> TimeoutError:
    ids = ", ".join(str(f.id) for f in pending[:5])
    more = f" (+{len(pending) - 5} more)" if len(pending) > 5 else ""
    return TimeoutError(f"still waiting for future(s) {ids}{more}")


def _wait_all(t: Transport, leaves: list[APIFuture[Any]], timeout: float | None, method: str) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout
    pending = list(leaves)
    while pending:
        wait = _wait_s(deadline)
        if len(pending) == 1:
            f = pending[0]
            state = t.request(
                "GET",
                f"/v1/futures/{f.id}",
                params={"wait_s": wait},
                method_name=method,
                timeout=t.timeout + wait,
            )
            if f._settle(state):
                pending = []
        else:
            for chunk in _chunks(pending, MAX_FUTURES_PER_RETRIEVE):
                states = t.request(
                    "POST",
                    "/v1/futures/retrieve",
                    json={"ids": [f.id for f in chunk]},
                    params={"wait_s": wait},
                    method_name=method,
                    timeout=t.timeout + wait,
                )
                for f, state in zip(chunk, states, strict=False):
                    f._settle(state, "POST /v1/futures/retrieve")
            pending = [f for f in pending if not f.done()]
        if pending and deadline is not None and time.monotonic() >= deadline:
            raise _timeout_error(pending)


async def _wait_all_async(t: Transport, leaves: list[APIFuture[Any]], timeout: float | None, method: str) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout
    pending = list(leaves)
    while pending:
        wait = _wait_s(deadline)
        if len(pending) == 1:
            f = pending[0]
            state = await t.arequest(
                "GET",
                f"/v1/futures/{f.id}",
                params={"wait_s": wait},
                method_name=method,
                timeout=t.timeout + wait,
            )
            if f._settle(state):
                pending = []
        else:
            for chunk in _chunks(pending, MAX_FUTURES_PER_RETRIEVE):
                states = await t.arequest(
                    "POST",
                    "/v1/futures/retrieve",
                    json={"ids": [f.id for f in chunk]},
                    params={"wait_s": wait},
                    method_name=method,
                    timeout=t.timeout + wait,
                )
                for f, state in zip(chunk, states, strict=False):
                    f._settle(state, "POST /v1/futures/retrieve")
            pending = [f for f in pending if not f.done()]
        if pending and deadline is not None and time.monotonic() >= deadline:
            raise _timeout_error(pending)


def _chunks(items: list[APIFuture[Any]], n: int) -> list[list[APIFuture[Any]]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


# ---------------------------------------------------------------- gather


def gather(*futures: APIFuture[Any], timeout: float | None = None) -> list[Any]:
    """Wait for several futures with one long-poll per round trip; returns their results in order.

    Raises the first ``OperationFailedError`` in order once every future has settled.
    """
    leaves = [leaf for f in futures for leaf in f.leaves()]
    if leaves:
        t = leaves[0]._t
        assert t is not None
        _wait_all(t, leaves, timeout, "gather")
    return [f._finish() for f in futures]


async def gather_async(*futures: APIFuture[Any], timeout: float | None = None) -> list[Any]:
    """``gather()`` for async code."""
    leaves = [leaf for f in futures for leaf in f.leaves()]
    if leaves:
        t = leaves[0]._t
        assert t is not None
        await _wait_all_async(t, leaves, timeout, "gather")
    return [f._finish() for f in futures]


# ---------------------------------------------------------------- steps in flight


class Steps:
    """Training steps sent before the last one's results are read: up to ``in_flight`` at once, so the trainer has the
    next step queued when it finishes one, instead of waiting for your round trip.

        steps = endor.Steps()                       # 2 steps in flight
        for batch in batches:
            for out, opt in steps.add(run.forward_backward(batch), run.optim_step(learning_rate=1e-4)):
                print(opt.step, out.loss)
        for out, opt in steps.wait():               # the steps still in flight
            print(opt.step, out.loss)

    ``add`` takes one step's futures, already sent, and returns the results of the steps it waited for, oldest first,
    each as ``gather`` returns them: none while fewer than ``in_flight`` steps are out, then the oldest step's.
    ``wait`` reads every step still in flight. A failed step raises its ``OperationFailedError`` from the ``add`` or
    ``wait`` that reads it; the steps sent after it stay in flight. A call you send between two ``add`` calls
    (``run.forward`` on held-out data, ``save_checkpoint``) runs after every step added so far, as the run executes
    calls in order. ``in_flight=1`` waits for each step before the next one is sent.
    """

    def __init__(self, in_flight: int = 2) -> None:
        if in_flight < 1:
            raise ValueError("in_flight must be at least 1")
        self.in_flight = in_flight
        self._sent: deque[tuple[APIFuture[Any], ...]] = deque()

    def __len__(self) -> int:
        """The steps added and not read yet."""
        return len(self._sent)

    def add(self, *futures: APIFuture[Any]) -> list[list[Any]]:
        """Add a step (its futures, already sent); returns the results of the steps it read."""
        self._sent.append(futures)
        return [gather(*self._sent.popleft()) for _ in range(len(self._sent) - self.in_flight + 1)]

    def wait(self) -> list[list[Any]]:
        """Read every step still in flight; returns their results, oldest first."""
        return [gather(*self._sent.popleft()) for _ in range(len(self._sent))]

    async def add_async(self, *futures: APIFuture[Any]) -> list[list[Any]]:
        """``add`` for async code."""
        self._sent.append(futures)
        return [await gather_async(*self._sent.popleft()) for _ in range(len(self._sent) - self.in_flight + 1)]

    async def wait_async(self) -> list[list[Any]]:
        """``wait`` for async code."""
        return [await gather_async(*self._sent.popleft()) for _ in range(len(self._sent))]
