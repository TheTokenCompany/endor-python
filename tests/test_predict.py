"""client.predict: the answers to every question of every row, in the rows' order, from one server-side evaluation of
the rows without their labels (its progress printed while it runs, its results file fetched from storage without the
API key), for a custom model or a base model; and client.evaluations.create on rows sent inline."""

from __future__ import annotations

import pytest

import endor
import endor._predict
from endor import Choice, ChoiceAnswer, NotFoundError, NoulAnswer, OperationFailedError

from .conftest import URGENT, requests_to, rows
from .fake_api import FakeEndor


def tickets(n: int) -> list[dict]:
    """`n` rows whose choice options are named after the row (so each row's answer is its own) and that label
    only the yes/no question."""
    return [
        {
            "id": "same",  # ids needn't be unique: answers come back by position
            "state": f"ticket {i}",
            "questions": {"dept": Choice(criteria={f"d{i}": None, "other": None}), "urgent": URGENT},
            "labels": {"urgent": True},
        }
        for i in range(n)
    ]


def test_predict_returns_every_rows_answers_in_order(client: endor.EndorClient, fake: FakeEndor) -> None:
    out = client.predict("jev-9b", tickets(5), progress=False)
    assert [a["dept"].choice for a in out] == [f"d{i}" for i in range(5)]  # the fake's lines come in reverse
    assert all(isinstance(a["dept"], ChoiceAnswer) and isinstance(a["urgent"], NoulAnswer) for a in out)
    (req,) = requests_to(fake, "/v1/evaluations", "POST")
    assert req.body["model"] == "jev-9b" and "dataset" not in req.body
    assert len(req.body["rows"]) == 5 and all("labels" not in r for r in req.body["rows"])  # every question answered
    assert req.headers["x-endor-sdk-method"] == "client.predict" and req.headers.get("idempotency-key")
    (download,) = fake.file_requests
    assert "authorization" not in download.headers and not any(h.startswith("x-endor") for h in download.headers)


async def test_predict_async(client: endor.EndorClient, fake: FakeEndor) -> None:
    out = await client.predict_async("jev-9b", tickets(3), progress=False)  # a base model, untuned
    assert [a["dept"].choice for a in out] == ["d0", "d1", "d2"] and set(out[0]) == {"dept", "urgent"}


def test_predict_prints_progress_while_it_runs(
    client: endor.EndorClient,
    fake: FakeEndor,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(endor._predict, "PROGRESS_EVERY_S", 0.0)
    fake.future_polls = 2
    client.predict("jev-9b", tickets(4))
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2 and all(x.startswith("evaluation evl_") and "2/4 rows (50%)" in x for x in lines)
    fake.future_polls = 2
    client.predict("jev-9b", tickets(4), progress=False)
    assert capsys.readouterr().err == ""


def test_predict_waits_at_most_its_timeout(client: endor.EndorClient, fake: FakeEndor) -> None:
    fake.future_polls = 10
    with pytest.raises(TimeoutError):
        client.predict("jev-9b", tickets(2), timeout=0, progress=False)


def test_predict_errors(client: endor.EndorClient, fake: FakeEndor) -> None:
    with pytest.raises(ValueError):
        client.predict("jev-9b", [])
    with pytest.raises(NotFoundError):
        client.predict("nope", tickets(1), progress=False)
    real = fake.future

    def failing(kind: str, result: dict | None = None, **kw: object) -> dict:
        return real(kind, result, error={"code": "context_too_long", "message": "Too long."})

    fake.future = failing  # type: ignore[method-assign]
    with pytest.raises(OperationFailedError) as e:
        client.predict("jev-9b", tickets(1), progress=False)
    assert e.value.code == "context_too_long"


def test_predict_checks_the_row_limit_locally(
    client: endor.EndorClient, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(endor._constants, "MAX_UPLOAD_ROWS", 2)
    with pytest.raises(ValueError, match="at most 2 rows"):
        client.predict("jev-9b", tickets(3))
    assert not requests_to(fake, "/evaluations", "POST")


def test_evaluate_takes_rows_inline(client: endor.EndorClient, fake: FakeEndor) -> None:
    ev = client.evaluations.create("jev-9b", rows=rows(6)).result()
    assert ev.status == "completed" and ev.dataset is None and ev.results and ev.results["n"] == 12
    assert (ev.rows_total, ev.rows_done) == (6, 6)
    (req,) = requests_to(fake, "/evaluations", "POST")
    assert "dataset" not in req.body and req.body["rows"][0]["labels"] == {"dept": "tech", "urgent": False}
    with pytest.raises(ValueError):
        client.evaluations.create("jev-9b")  # a dataset or rows
    with pytest.raises(ValueError):
        client.evaluations.create("jev-9b", "heldout", rows=rows(1))
