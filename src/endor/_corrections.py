"""Corrections of decisions (``client.corrections``): for the questions you checked, whether the model answered right
and, if not, the right answer. Continuous learning projects learn from them.

    res = client.system_one(state, questions, model="tickets")
    client.corrections.create(res, {
        "tone": {"correct": True},                         # the model was right
        "dept": {"correct": False, "answer": "refunds"},   # wrong, and the right answer
        "urgent": {"correct": False},                      # wrong (a yes/no question: so the other answer)
    }, comment="the customer only asked a question")
    client.corrections.get(res)                            # the decision's correction
    client.corrections.delete(res)                         # withdraw it
    client.corrections.batch([{"request_id": "req_...", "questions": {"urgent": {"correct": True}}}])

A decision has one correction: creating it again replaces it whole, and the same questions and comment again change
nothing. Corrections are free.

Endor records a decision just after it answers, so a correction sent at once can find no decision yet (404
``not_found`` naming it). ``create`` and ``batch`` try again after each of ``DECISION_RETRY_DELAYS`` (5 s in all), then
raise ``NotFoundError`` as for any unknown decision.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping
from typing import Any

from ._http import Transport
from ._log import logger
from ._parse import parse
from .errors import EndorError, NotFoundError
from .types import CorrectionInfo, QuestionCorrection, SystemOneResponse

__all__ = ["Corrections", "MAX_ITEMS", "DECISION_RETRY_DELAYS"]

MAX_ITEMS = 1000
"""Items per ``batch`` call; the API keeps all of them or none."""

DECISION_RETRY_DELAYS = (0.25, 0.5, 1.0, 1.25, 2.0)
"""Seconds to wait before each new try of a correction whose decision Endor hasn't recorded yet (5 s in all)."""


class Corrections:
    """``client.corrections``: correct decisions, read and withdraw their corrections. A decision is its
    ``SystemOneResponse`` or its id (``res.id``, ``req_...``, the same as its ``X-Request-Id``)."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def create(
        self,
        request_id: str | SystemOneResponse,
        questions: Mapping[str, QuestionCorrection],
        *,
        comment: str | None = None,
    ) -> CorrectionInfo:
        """Correct a decision (``POST /v1/corrections``), in any project.

        Args:
            request_id: The decision: its ``SystemOneResponse`` or its id.
            questions: One or more of its questions, by name, each ``{"correct": True}`` (the model was right),
                ``{"correct": False, "answer": ...}`` (the right answer) or ``{"correct": False}`` (only that it was
                wrong: for a noul, or a choice or score of two options, that names the right answer; with more, the
                project never learns the wrong answer, and nothing else). An ``answer`` must be exactly one of the
                question's options as the decision asked it: a choice's option id (case and spelling included), a
                noul's ``True`` or ``False``, a score's level index as an ``int`` (0 the first). Anything else, an
                "other" answer included, is refused; when none of the options fits, send ``{"correct": False}``.
            comment: Anything that helps, such as what was misread (at most 1,000 characters).

        Returns:
            The decision's correction, each question with ``served`` (what Endor answered) and ``answer`` (the
            right answer, ``None`` when only known to be wrong).

        Raises:
            NotFoundError: Not an answered decision of your organization. Endor records a decision just after it
                answers, so a correction sent at once can find none yet: the SDK tries again for up to 5 s
                (``DECISION_RETRY_DELAYS``) before it raises.
            UnprocessableEntityError: A question the decision didn't ask (``param``: ``questions.<name>``), or an
                answer that isn't exactly one of its options or contradicts ``correct`` (``questions.<name>.answer``;
                the message lists the options).
        """
        r = self._post("/v1/corrections", _item(request_id, questions, comment), "client.corrections.create")
        return parse(CorrectionInfo, r, "POST /v1/corrections")

    async def create_async(
        self,
        request_id: str | SystemOneResponse,
        questions: Mapping[str, QuestionCorrection],
        *,
        comment: str | None = None,
    ) -> CorrectionInfo:
        """``create`` for async code."""
        body = _item(request_id, questions, comment)
        r = await self._post_async("/v1/corrections", body, "client.corrections.create")
        return parse(CorrectionInfo, r, "POST /v1/corrections")

    def batch(self, items: Iterable[Mapping[str, Any]]) -> list[CorrectionInfo]:
        """Correct many decisions in one call (``POST /v1/corrections/batch``), all or nothing: if one item fails,
        none is kept. For a backfill of decisions you checked earlier.

        Args:
            items: 1 to 1,000 items, each ``{"request_id", "questions", "comment"?}`` as for ``create``
                (``request_id`` may be a ``SystemOneResponse`` too).

        Returns:
            Each item's correction, in order.

        Raises:
            ValueError: No items, or more than 1,000.
            NotFoundError: An item's decision isn't one (``param``: ``items[i].request_id``), after trying again for
                up to 5 s as ``create`` does.
            UnprocessableEntityError: An item's question doesn't fit (``param``: ``items[i].questions.<name>``...).
        """
        r = self._post("/v1/corrections/batch", _items(items), "client.corrections.batch")
        return [parse(CorrectionInfo, x, "POST /v1/corrections/batch") for x in r.get("items", [])]

    async def batch_async(self, items: Iterable[Mapping[str, Any]]) -> list[CorrectionInfo]:
        """``batch`` for async code."""
        r = await self._post_async("/v1/corrections/batch", _items(items), "client.corrections.batch")
        return [parse(CorrectionInfo, x, "POST /v1/corrections/batch") for x in r.get("items", [])]

    def get(self, request_id: str | SystemOneResponse) -> CorrectionInfo:
        """The decision's correction. ``NotFoundError`` as for ``create``, and when the decision has none."""
        path = _path(request_id)
        return parse(CorrectionInfo, self._t.request("GET", path, method_name="client.corrections.get"), f"GET {path}")

    async def get_async(self, request_id: str | SystemOneResponse) -> CorrectionInfo:
        """``get`` for async code."""
        path = _path(request_id)
        r = await self._t.arequest("GET", path, method_name="client.corrections.get")
        return parse(CorrectionInfo, r, f"GET {path}")

    def delete(self, request_id: str | SystemOneResponse) -> None:
        """Withdraw the decision's correction: learning stops using it. ``NotFoundError`` as for ``get``."""
        self._t.request("DELETE", _path(request_id), method_name="client.corrections.delete")

    async def delete_async(self, request_id: str | SystemOneResponse) -> None:
        """``delete`` for async code."""
        await self._t.arequest("DELETE", _path(request_id), method_name="client.corrections.delete")

    def _post(self, path: str, body: dict[str, Any], method_name: str) -> Any:
        """POST a correction (or a batch), trying again while Endor hasn't recorded its decision yet."""
        for delay in DECISION_RETRY_DELAYS:
            try:
                return self._t.request("POST", path, json=body, method_name=method_name, idempotent=True)
            except NotFoundError as e:
                if not _decision_not_recorded(e, body):
                    raise
            time.sleep(delay)
        return self._t.request("POST", path, json=body, method_name=method_name, idempotent=True)

    async def _post_async(self, path: str, body: dict[str, Any], method_name: str) -> Any:
        """``_post`` for async code."""
        for delay in DECISION_RETRY_DELAYS:
            try:
                return await self._t.arequest("POST", path, json=body, method_name=method_name, idempotent=True)
            except NotFoundError as e:
                if not _decision_not_recorded(e, body):
                    raise
            await asyncio.sleep(delay)
        return await self._t.arequest("POST", path, json=body, method_name=method_name, idempotent=True)


def _decision_not_recorded(e: NotFoundError, body: dict[str, Any]) -> bool:
    """Whether ``e`` is the API's 404 ``not_found`` for a decision of this correction or batch (``No decision
    'req_...'.``): possibly one Endor hasn't recorded yet. Logs that the SDK tries again."""
    ids = [it["request_id"] for it in body.get("items", [body])]
    rid = next((i for i in ids if i and i in e.message), None)
    if e.code != "not_found" or "decision" not in e.message.lower() or rid is None:
        return False
    logger.info("decision %s isn't recorded yet (request %s); trying the correction again", rid, e.request_id)
    return True


def _request_id(decision: str | SystemOneResponse) -> str:
    if isinstance(decision, SystemOneResponse):
        if decision.id is None:
            raise EndorError("this response has no decision id: the server didn't send one (an older Endor API?)")
        return decision.id
    return decision


def _path(decision: str | SystemOneResponse) -> str:
    return f"/v1/corrections/{_request_id(decision)}"


def _questions(questions: Mapping[str, QuestionCorrection]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, q in questions.items():
        if not isinstance(q, Mapping) or not isinstance(q.get("correct"), bool):
            raise ValueError(
                f'questions[{name!r}]: send {{"correct": True}}, {{"correct": False, "answer": ...}} or '
                f'{{"correct": False}}, got {q!r}'
            )
        out[name] = dict(q)
    return out


def _item(
    request_id: str | SystemOneResponse, questions: Mapping[str, QuestionCorrection], comment: str | None
) -> dict[str, Any]:
    body: dict[str, Any] = {"request_id": _request_id(request_id), "questions": _questions(questions)}
    if comment is not None:
        body["comment"] = comment
    return body


def _items(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    out = [_item(it.get("request_id", ""), it.get("questions") or {}, it.get("comment")) for it in items]
    if not 1 <= len(out) <= MAX_ITEMS:
        raise ValueError(f"send 1 to {MAX_ITEMS:,} items per call, got {len(out):,}")
    return {"items": out}
