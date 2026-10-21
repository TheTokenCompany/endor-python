"""Corrections of decisions (``client.corrections``): a decision's response carries its id, and a correction is
created, read, replaced, withdrawn and sent in batches against it."""

from __future__ import annotations

import pytest

import endor
from endor import Choice, CorrectionInfo, SystemOneResponse

from .conftest import ANGER, DEPT, URGENT, requests_to
from .fake_api import FakeEndor

ROUTE = Choice(instructions="Route the ticket", criteria={"billing": "money", "tech": None, "sales": None})
QUESTIONS = {"dept": ROUTE, "pair": DEPT, "urgent": URGENT, "anger": ANGER}


def decide(client: endor.EndorClient, decider: str) -> SystemOneResponse:
    return client.system_one({"body": "charged twice"}, QUESTIONS, model=decider)


def test_a_decision_answers_with_its_id(client: endor.EndorClient, decider: str) -> None:
    res = decide(client, decider)
    assert res.id is not None and res.id.startswith("req_") and res.id == client._t.last.request_id


def test_create_get_replace_and_delete(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    res = decide(client, decider)
    # the fake answers uniformly: the choice's first option, a noul at 0.5 (true), the score's first level
    c = client.corrections.create(
        res,
        {
            "dept": {"correct": False, "answer": "tech"},
            "pair": {"correct": False},  # wrong, of two options: so the other one
            "urgent": {"correct": False},
            "anger": {"correct": True},
        },
        comment="misread",
    )
    assert isinstance(c, CorrectionInfo)
    assert (c.request_id, c.project, c.comment) == (res.id, decider, "misread")
    got = {k: (q.correct, q.answer, q.served) for k, q in c.questions.items()}
    assert got == {
        "dept": (False, "tech", "billing"),
        "pair": (False, "tech", "billing"),
        "urgent": (False, False, True),
        "anger": (True, 0, 0),
    }
    assert type(c.questions["urgent"].answer) is bool and type(c.questions["anger"].served) is int
    sent = requests_to(fake, "/v1/corrections", "POST")[0]
    assert sent.body == {
        "request_id": res.id,
        "questions": {
            "dept": {"correct": False, "answer": "tech"},
            "pair": {"correct": False},
            "urgent": {"correct": False},
            "anger": {"correct": True},
        },
        "comment": "misread",
    }
    assert sent.headers["x-endor-sdk-method"] == "client.corrections.create" and sent.headers.get("idempotency-key")

    assert client.corrections.get(res) == c
    assert res.id
    wrong = client.corrections.create(res.id, {"dept": {"correct": False}})  # 3 options: only known to be wrong
    assert "comment" not in requests_to(fake, "/v1/corrections", "POST")[-1].body
    assert wrong.questions["dept"].answer is None and list(wrong.questions) == ["dept"]  # replaced whole
    assert client.corrections.create(res, {"dept": {"correct": False}}) == wrong  # the same again: unchanged
    assert client.corrections.get(res.id) == wrong

    client.corrections.delete(res)
    assert requests_to(fake, f"/v1/corrections/{res.id}", "DELETE")
    with pytest.raises(endor.NotFoundError):
        client.corrections.get(res)
    with pytest.raises(endor.NotFoundError):
        client.corrections.delete(res)


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("dept", "Billing"),  # another case
        ("dept", " billing"),  # surrounding spaces
        ("dept", "money"),  # an option's description, not its key
        ("dept", "other"),  # not one of the options
        ("urgent", "true"),  # a noul takes true or false, not a string
        ("anger", "2"),  # a score takes its level index as a number
        ("anger", True),  # nor true for the level 1
    ],
)
def test_an_answer_must_be_exactly_one_of_the_options(
    client: endor.EndorClient, decider: str, question: str, answer: object
) -> None:
    res = decide(client, decider)
    with pytest.raises(endor.UnprocessableEntityError) as e:
        client.corrections.create(res, {question: {"correct": False, "answer": answer}})  # type: ignore[typeddict-item]
    assert e.value.param == f"questions.{question}.answer"


def test_errors(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    res = decide(client, decider)
    with pytest.raises(endor.UnprocessableEntityError) as e:
        client.corrections.create(res, {"dept": {"correct": True, "answer": "tech"}})  # served billing
    assert e.value.param == "questions.dept.answer"
    with pytest.raises(endor.UnprocessableEntityError) as e:
        client.corrections.create(res, {"tone": {"correct": True}})  # a question the decision didn't ask
    assert e.value.param == "questions.tone"
    with pytest.raises(endor.NotFoundError):
        client.corrections.create("req_unknown", {"urgent": {"correct": True}})
    with pytest.raises(endor.EndorError, match="no decision id"):
        client.corrections.create(SystemOneResponse(model=decider), {"urgent": {"correct": True}})
    with pytest.raises(ValueError, match='"correct": True'):
        client.corrections.create(res, {"urgent": True})  # type: ignore[dict-item]


def test_batch_is_all_or_nothing(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    a, b = decide(client, decider), decide(client, decider)
    items = [
        {"request_id": a, "questions": {"dept": {"correct": True}}},
        {"request_id": b.id, "questions": {"urgent": {"correct": False}}, "comment": "chargeback"},
    ]
    with pytest.raises(endor.UnprocessableEntityError) as e:
        client.corrections.batch(
            [*items, {"request_id": a.id, "questions": {"urgent": {"correct": False, "answer": 1}}}]
        )
    assert e.value.param == "items[2].questions.urgent.answer"
    assert fake.corrections == {}
    got = client.corrections.batch(items)
    assert [(c.request_id, c.comment) for c in got] == [(a.id, None), (b.id, "chargeback")]
    sent = requests_to(fake, "/v1/corrections/batch", "POST")[-1]
    assert sent.body["items"][0]["request_id"] == a.id and sent.headers.get("idempotency-key")
    for wrong in ([], [items[0]] * 1001):
        with pytest.raises(ValueError, match="1 to 1,000 items"):
            client.corrections.batch(wrong)


async def test_async(client: endor.EndorClient, fake: FakeEndor, decider: str) -> None:
    res = await client.system_one_async({"body": "charged twice"}, QUESTIONS, model=decider)
    c = await client.corrections.create_async(res, {"urgent": {"correct": False}})
    assert c.questions["urgent"].answer is False
    assert await client.corrections.get_async(res) == c
    many = await client.corrections.batch_async([{"request_id": res, "questions": {"anger": {"correct": True}}}])
    assert [list(x.questions) for x in many] == [["anger"]]
    await client.corrections.delete_async(res)
    with pytest.raises(endor.NotFoundError):
        await client.corrections.get_async(res)
