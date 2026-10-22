"""Corrections of decisions (``client.corrections``): a decision's response carries its id, and a correction is
created, read, replaced, withdrawn and sent in batches against it."""

from __future__ import annotations

import pytest

import endor
from endor import Choice, CorrectionInfo, SystemOneResponse, _corrections

from .conftest import ANGER, DEPT, URGENT, requests_to
from .fake_api import FakeEndor, HTTPError

ROUTE = Choice(instructions="Route the ticket", criteria={"billing": "money", "tech": None, "sales": None})
QUESTIONS = {"dept": ROUTE, "pair": DEPT, "urgent": URGENT, "anger": ANGER}
WAITS = _corrections.DECISION_RETRY_DELAYS


@pytest.fixture(autouse=True)
def _no_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown decision is tried again as one not recorded yet: without the waits here."""
    monkeypatch.setattr(_corrections, "DECISION_RETRY_DELAYS", (0.0,) * len(WAITS))


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


def test_a_correction_right_after_its_decision_waits_for_it(
    client: endor.EndorClient, fake: FakeEndor, decider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Endor records a decision just after it answers, so a correction sent at once can find none yet (404
    not_found naming it): the SDK tries again for up to 5 s, then raises as for any unknown decision."""
    waited: list[float] = []
    monkeypatch.setattr(_corrections, "DECISION_RETRY_DELAYS", WAITS)
    monkeypatch.setattr(_corrections.time, "sleep", waited.append)
    assert sum(WAITS) == 5.0
    fake.decision_record_lag = 2  # the next decisions are recorded after two tries
    res = decide(client, decider)
    assert client.corrections.create(res, {"urgent": {"correct": True}}).request_id == res.id
    assert waited == list(WAITS[:2]) and len(requests_to(fake, "/v1/corrections", "POST")) == 3
    other = decide(client, decider)
    waited.clear()
    got = client.corrections.batch(
        [
            {"request_id": res, "questions": {"dept": {"correct": True}}},
            {"request_id": other, "questions": {"dept": {"correct": True}}},
        ]
    )
    assert [c.request_id for c in got] == [res.id, other.id] and waited == list(WAITS[:2])

    fake.decision_record_lag = len(WAITS) + 1  # never in time: the last try's 404 is raised
    late = decide(client, decider)
    waited.clear()
    with pytest.raises(endor.NotFoundError) as e:
        client.corrections.create(late, {"urgent": {"correct": True}})
    assert e.value.code == "not_found" and late.id and late.id in e.value.message and waited == list(WAITS)

    fake.decision_record_lag = 0
    now = decide(client, decider)
    waited.clear()
    sent = len(requests_to(fake, "/v1/corrections", "POST"))
    with pytest.raises(endor.UnprocessableEntityError):  # anything else is raised at once
        client.corrections.create(now, {"tone": {"correct": True}})
    fake.fail_next.append(HTTPError(404, "not_found", "Not Found"))  # a 404 that doesn't name the decision
    with pytest.raises(endor.NotFoundError):
        client.corrections.create(now, {"urgent": {"correct": True}})
    assert waited == [] and len(requests_to(fake, "/v1/corrections", "POST")) == sent + 2


async def test_async_corrections_wait_for_the_decision_too(
    client: endor.EndorClient, fake: FakeEndor, decider: str
) -> None:
    fake.decision_record_lag = 1
    res = await client.system_one_async({"body": "charged twice"}, QUESTIONS, model=decider)
    assert (await client.corrections.create_async(res, {"urgent": {"correct": True}})).request_id == res.id
    other = await client.system_one_async({"body": "charged twice"}, QUESTIONS, model=decider)
    many = await client.corrections.batch_async([{"request_id": other, "questions": {"anger": {"correct": True}}}])
    assert [c.request_id for c in many] == [other.id]
    assert len(requests_to(fake, "/v1/corrections", "POST")) == 2
    assert len(requests_to(fake, "/v1/corrections/batch", "POST")) == 2
