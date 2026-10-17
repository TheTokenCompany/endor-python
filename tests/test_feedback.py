"""Feedback on decisions (``client.feedback``): a decision's response carries its id, and feedback is sent, listed,
replaced, withdrawn and sent in bulk against it."""

from __future__ import annotations

import pytest

import endor
from endor import FeedbackInfo, SystemOneResponse

from .conftest import ANGER, DEPT, URGENT, requests_to
from .fake_api import FakeEndor

QUESTIONS = {"dept": DEPT, "urgent": URGENT, "anger": ANGER}


def decide(client: endor.EndorClient, decider: str) -> SystemOneResponse:
    return client.system_one({"body": "charged twice"}, QUESTIONS, model=decider)


def test_a_decision_answers_with_its_id(client: endor.EndorClient, decider: str) -> None:
    res = decide(client, decider)
    assert res.id is not None and res.id.startswith("req_") and res.id == client._t.last.request_id


def test_send_list_replace_and_delete(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    res = decide(client, decider)
    fb = client.feedback.send(res, {"dept": "tech", "urgent": True, "anger": 2}, source="review", comment="misread")
    assert isinstance(fb, FeedbackInfo)
    assert (fb.decision, fb.project, fb.source, fb.comment) == (res.id, decider, "review", "misread")
    # the fake answers uniformly: the choice's first option, a noul at 0.5, the score's first level
    assert fb.answers == {"dept": "tech", "urgent": True, "anger": 2}
    assert fb.served == {"dept": "billing", "urgent": True, "anger": 0}
    assert type(fb.served["urgent"]) is bool and type(fb.served["anger"]) is int
    sent = requests_to(fake, f"/v1/decisions/{res.id}/feedback", "PUT")[0]
    assert sent.body == {
        "source": "review",
        "answers": {"dept": "tech", "urgent": True, "anger": 2},
        "comment": "misread",
    }
    assert sent.headers["x-endor-sdk-method"] == "client.feedback.send"

    assert res.id
    user = client.feedback.send(res.id, {"urgent": False}, source="user")
    assert "comment" not in requests_to(fake, "/feedback", "PUT")[-1].body
    assert [f.source for f in client.feedback.list(res)] == ["review", "user"]
    again = client.feedback.send(res, {"urgent": False}, source="user")
    assert again == user  # the same answers and comment: unchanged
    replaced = client.feedback.send(res, {"urgent": True}, source="user")
    assert [f.answers for f in client.feedback.list(res)] == [fb.answers, replaced.answers]

    client.feedback.delete(res, source="review")
    assert requests_to(fake, "/feedback", "DELETE")[-1].params == {"source": "review"}
    assert [f.source for f in client.feedback.list(res)] == ["user"]
    with pytest.raises(endor.NotFoundError):
        client.feedback.delete(res, source="review")


def test_errors_name_the_answer(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    res = decide(client, decider)
    with pytest.raises(endor.UnprocessableEntityError) as e:
        client.feedback.send(res, {"anger": "2"}, source="review")
    assert e.value.param == "answers.anger"
    with pytest.raises(endor.NotFoundError):
        client.feedback.send("req_unknown", {"urgent": True}, source="review")
    with pytest.raises(endor.EndorError, match="no decision id"):
        client.feedback.send(SystemOneResponse(model=decider), {"urgent": True}, source="review")


def test_send_many_is_all_or_nothing(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    a, b = decide(client, decider), decide(client, decider)
    items = [
        {"decision": a, "source": "review", "answers": {"dept": "billing"}},
        {"decision": b.id, "source": "outcome", "answers": {"urgent": False}, "comment": "chargeback"},
    ]
    with pytest.raises(endor.UnprocessableEntityError) as e:
        client.feedback.send_many([*items, {"decision": a.id, "source": "user", "answers": {"urgent": "yes"}}])
    assert e.value.param == "items[2].answers.urgent"
    assert client.feedback.list(a) == client.feedback.list(b) == []
    got = client.feedback.send_many(items)
    assert [(f.decision, f.source, f.comment) for f in got] == [(a.id, "review", None), (b.id, "outcome", "chargeback")]
    sent = requests_to(fake, "/v1/feedback", "POST")[-1]
    assert sent.body["items"][0]["decision"] == a.id and sent.headers.get("idempotency-key")
    for wrong in ([], [items[0]] * 1001):
        with pytest.raises(ValueError, match="1 to 1,000 items"):
            client.feedback.send_many(wrong)


async def test_async(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    res = await client.system_one_async({"body": "charged twice"}, QUESTIONS, model=decider)
    fb = await client.feedback.send_async(res, {"urgent": False}, source="outcome")
    assert fb.served == {"urgent": True}
    assert await client.feedback.list_async(res) == [fb]
    many = await client.feedback.send_many_async([{"decision": res, "source": "review", "answers": {"anger": 1}}])
    assert [f.source for f in many] == ["review"]
    await client.feedback.delete_async(res, source="outcome")
    assert [f.source for f in await client.feedback.list_async(res)] == ["review"]
