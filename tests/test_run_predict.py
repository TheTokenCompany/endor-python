"""run.predict: decision rows answered with the run's current adapter, before saving."""

from __future__ import annotations

import pytest

import endor
from endor import ChoiceAnswer, DecisionRow, NoulAnswer, ScoreAnswer

from .conftest import ANGER, DEPT, URGENT, rows
from .fake_api import FakeEndor


def posts(fake: FakeEndor, op: str) -> list[dict]:
    return [r.body for r in fake.requests if r.method == "POST" and r.path.endswith(f"/{op}")]


def test_predict_answers_each_row_in_order_as_a_decision_does(
    client: endor.EndorClient, project: endor.Project, fake: FakeEndor
) -> None:
    mixed = [
        *rows(2),  # dataset rows: their ids and labels are not sent
        {"state": "where is my parcel?", "questions": {"anger": ANGER}},
        DecisionRow(state={"body": "refund"}, questions={"urgent": URGENT, "dept": DEPT}),
    ]
    with project.runs.create("jev-9b") as run:
        answers = run.predict(mixed).result()
        assert [list(a) for a in answers] == [["dept", "urgent"], ["dept", "urgent"], ["anger"], ["urgent", "dept"]]
        assert isinstance(answers[0]["dept"], ChoiceAnswer) and isinstance(answers[0]["urgent"], NoulAnswer)
        assert isinstance(answers[2]["anger"], ScoreAnswer)
        for row, got in zip(map(endor.data.to_row, mixed), answers, strict=True):
            decided = client.system_one(row.state, row.questions, model=f"{project.name}/jev-9b")
            assert got == decided.answers
        (body,) = posts(fake, "predict")
        assert body["seq_id"] == 0 and set(body) == {"rows", "seq_id"}
        assert body["rows"][0] == {
            "state": {"body": "ticket 0"},
            "questions": {"dept": endor.types.question_dict(DEPT), "urgent": endor.types.question_dict(URGENT)},
        }
        assert run.next_seq_id == 1


def test_predict_runs_in_order_with_the_training_steps(project: endor.Project, fake: FakeEndor) -> None:
    with project.runs.create("jev-9b") as run:
        fb = run.forward_backward(endor.data.rows_to_datums(rows(2)))
        scored = run.predict(rows(3))
        opt = run.optim_step(learning_rate=1e-4)
        out, answers, step = endor.gather(fb, scored, opt)
        assert out.metrics["n"] == 4 and len(answers) == 3 and step.step == 1
        assert [
            b["seq_id"] for b in posts(fake, "forward_backward") + posts(fake, "predict") + posts(fake, "optim_step")
        ] == [0, 1, 2]


def test_long_lists_are_split_by_questions_and_merged_in_order(
    project: endor.Project, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(endor.runs, "MAX_PREDICT_QUESTIONS", 5)
    many = [{"state": f"s{i}", "questions": {f"q{i}": URGENT, "dept": DEPT}} for i in range(6)]  # 12 questions
    with project.runs.create("jev-9b") as run:
        fut = run.predict(many)
        assert fut.id is None and len(fut.leaves()) == 3
        answers = fut.result()
        assert [list(a) for a in answers] == [[f"q{i}", "dept"] for i in range(6)]
        assert [len(b["rows"]) for b in posts(fake, "predict")] == [2, 2, 2]  # 4 questions each, at most 5
        assert [b["seq_id"] for b in posts(fake, "predict")] == [0, 1, 2]


def test_rows_are_checked_before_anything_is_sent(project: endor.Project, fake: FakeEndor) -> None:
    with project.runs.create("jev-9b") as run:
        too_many = {"state": "x", "questions": {f"q{i}": URGENT for i in range(65)}}
        with pytest.raises(ValueError, match=r"rows\[1\]: at most 64 questions"):
            run.predict([rows(1)[0], too_many])
        with pytest.raises(ValueError, match="no rows"):
            run.predict([])
        with pytest.raises(ValueError, match="unrecognized row"):
            run.predict([{"state": "x"}])
        assert posts(fake, "predict") == [] and run.next_seq_id == 0


async def test_predict_async(project: endor.Project, fake: FakeEndor) -> None:
    async with project.runs.create("jev-9b") as run:
        fut = await run.predict_async(rows(2))
        answers = await fut
        assert len(answers) == 2 and isinstance(answers[1]["urgent"], NoulAnswer)
        assert [b["seq_id"] for b in posts(fake, "predict")] == [0]
