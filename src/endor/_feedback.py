"""Feedback on decisions (``client.feedback``): the right answers to some of a past decision's questions, once you
know them.

    res = client.system_one(state, questions, model="loans")
    client.feedback.send(res, {"approve": False, "risk": 3}, source="review", comment="income was misread")
    client.feedback.list(res)                            # the feedback of each source
    client.feedback.delete(res, source="review")
    client.feedback.send_many([{"decision": "req_...", "source": "outcome", "answers": {"approve": False}}])

Each source has one feedback per decision: sending again replaces it, and the same answers and comment again change
nothing. Feedback is free. Managed projects learn from feedback.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping
from typing import Any

from ._http import Transport
from .errors import EndorError
from .types import FeedbackAnswer, FeedbackInfo, FeedbackSource, SystemOneResponse

__all__ = ["Feedback", "MAX_ITEMS"]

MAX_ITEMS = 1000
"""Items per ``send_many`` call; the API keeps all of them or none."""


class Feedback:
    """``client.feedback``: send, read and withdraw feedback on decisions. A decision is its ``SystemOneResponse`` or
    its id (``res.id``, ``req_...``)."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def send(
        self,
        decision: str | SystemOneResponse,
        answers: Mapping[str, FeedbackAnswer],
        *,
        source: FeedbackSource,
        comment: str | None = None,
    ) -> FeedbackInfo:
        """Tell Endor the right answers to some of a decision's questions (``PUT
        /v1/decisions/{decision_id}/feedback``).

        Args:
            decision: The decision: its ``SystemOneResponse`` or its id.
            answers: The right answer to one or more of its questions, by name: a choice's option id, a noul's
                ``True`` or ``False``, a score's level index (0 is the first level).
            source: ``"review"`` (someone on your team checked the decision), ``"outcome"`` (what actually happened,
                such as a chargeback) or ``"user"`` (an end user's opinion, not trusted).
            comment: Anything that helps, such as what was misread (at most 1,000 characters).

        Returns:
            The decision's feedback from ``source``, with ``served``: what Endor answered to the same questions.

        Raises:
            NotFoundError: Not an answered decision of your organization. Endor records a decision a moment after it
                answers, so feedback sent at once may need a retry.
            UnprocessableEntityError: A question the decision didn't ask, or an answer that doesn't fit it
                (``param``: ``answers.<name>``).
        """
        r = self._t.request(
            "PUT", _path(decision), json=_item(source, answers, comment), method_name="client.feedback.send"
        )
        return FeedbackInfo.model_validate(r)

    async def send_async(
        self,
        decision: str | SystemOneResponse,
        answers: Mapping[str, FeedbackAnswer],
        *,
        source: FeedbackSource,
        comment: str | None = None,
    ) -> FeedbackInfo:
        """``send`` for async code."""
        r = await self._t.arequest(
            "PUT", _path(decision), json=_item(source, answers, comment), method_name="client.feedback.send"
        )
        return FeedbackInfo.model_validate(r)

    def list(self, decision: str | SystemOneResponse) -> builtins.list[FeedbackInfo]:
        """The decision's feedback, one for each source that sent some (an empty list when none).
        ``NotFoundError`` as for ``send``."""
        r = self._t.request("GET", _path(decision), method_name="client.feedback.list")
        return [FeedbackInfo.model_validate(x) for x in r.get("items", [])]

    async def list_async(self, decision: str | SystemOneResponse) -> builtins.list[FeedbackInfo]:
        """``list`` for async code."""
        r = await self._t.arequest("GET", _path(decision), method_name="client.feedback.list")
        return [FeedbackInfo.model_validate(x) for x in r.get("items", [])]

    def delete(self, decision: str | SystemOneResponse, *, source: FeedbackSource) -> None:
        """Withdraw the decision's feedback from ``source``. ``NotFoundError`` as for ``send``, and when the decision
        has no feedback from ``source``."""
        self._t.request("DELETE", _path(decision), params={"source": source}, method_name="client.feedback.delete")

    async def delete_async(self, decision: str | SystemOneResponse, *, source: FeedbackSource) -> None:
        """``delete`` for async code."""
        await self._t.arequest(
            "DELETE", _path(decision), params={"source": source}, method_name="client.feedback.delete"
        )

    def send_many(self, items: Iterable[Mapping[str, Any]]) -> builtins.list[FeedbackInfo]:
        """Feedback on many decisions in one call (``POST /v1/feedback``), all or nothing: if one item fails, none is
        kept.

        Args:
            items: 1 to 1,000 items, each ``{"decision", "source", "answers", "comment"?}`` as for ``send``
                (``decision`` may be a ``SystemOneResponse`` too).

        Returns:
            Each item's feedback, in order.

        Raises:
            ValueError: No items, or more than 1,000.
            NotFoundError: An item's decision isn't one (``param``: ``items[i].decision``).
            UnprocessableEntityError: An item's answer doesn't fit (``param``: ``items[i].answers.<name>``).
        """
        r = self._t.request(
            "POST", "/v1/feedback", json=_items(items), method_name="client.feedback.send_many", idempotent=True
        )
        return [FeedbackInfo.model_validate(x) for x in r.get("items", [])]

    async def send_many_async(self, items: Iterable[Mapping[str, Any]]) -> builtins.list[FeedbackInfo]:
        """``send_many`` for async code."""
        r = await self._t.arequest(
            "POST", "/v1/feedback", json=_items(items), method_name="client.feedback.send_many", idempotent=True
        )
        return [FeedbackInfo.model_validate(x) for x in r.get("items", [])]


def _decision_id(decision: str | SystemOneResponse) -> str:
    if isinstance(decision, SystemOneResponse):
        if decision.id is None:
            raise EndorError("this response has no decision id: the server didn't send one (an older Endor API?)")
        return decision.id
    return decision


def _path(decision: str | SystemOneResponse) -> str:
    return f"/v1/decisions/{_decision_id(decision)}/feedback"


def _item(source: str, answers: Mapping[str, FeedbackAnswer], comment: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {"source": source, "answers": dict(answers)}
    if comment is not None:
        body["comment"] = comment
    return body


def _items(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    out = [dict(it) for it in items]
    if not 1 <= len(out) <= MAX_ITEMS:
        raise ValueError(f"send 1 to {MAX_ITEMS:,} items per call, got {len(out):,}")
    for it in out:
        if isinstance(it.get("decision"), SystemOneResponse):
            it["decision"] = _decision_id(it["decision"])
    return {"items": out}
