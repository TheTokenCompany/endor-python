from __future__ import annotations

import json
from pathlib import Path

import pytest

import endor
from endor import Choice, Datum, DecisionRow, Noul, Score, Target, data
from endor.types import answer_probabilities, option_keys, question_dict

from .conftest import ANGER, DEPT, URGENT, rows


class TestQuestions:
    def test_option_keys(self) -> None:
        assert option_keys(URGENT) == ["false", "true"]
        assert option_keys(DEPT) == ["billing", "tech"]
        assert option_keys(ANGER) == ["0", "1", "2"]
        assert option_keys({"type": "score", "criteria": ["a", "b"]}) == ["0", "1"]

    def test_question_dict(self) -> None:
        assert question_dict(Noul()) == {"type": "noul"}
        assert question_dict(Noul(instructions="x", criteria={"true": "yes", "false": None})) == {
            "type": "noul",
            "instructions": "x",
            "criteria": {"true": "yes", "false": None},
        }
        assert question_dict({"type": "noul"}) == {"type": "noul"}
        with pytest.raises(ValueError):
            question_dict({"type": "other"})
        with pytest.raises(ValueError):
            question_dict({"type": "score"})
        with pytest.raises(TypeError):
            question_dict("noul")  # type: ignore[arg-type]

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            Choice(criteria={})
        with pytest.raises(ValueError):
            Score(criteria=["only one"])
        with pytest.raises(ValueError):
            Score(criteria=[str(i) for i in range(11)])
        with pytest.raises(ValueError):
            Noul(unknown=1)  # type: ignore[call-arg]
        Choice(criteria={f"o{i}": None for i in range(255)})

    def test_answer_probabilities(self) -> None:
        assert answer_probabilities(endor.NoulAnswer(noul=0.2)) == {"false": 0.8, "true": 0.2}
        c = endor.ChoiceAnswer(choice="a", probabilities={"a": 0.6, "b": 0.4}, confidence=0.2)
        assert answer_probabilities(c) == {"a": 0.6, "b": 0.4}


class TestRows:
    def test_native_rows(self) -> None:
        r = data.to_row(rows(1)[0])
        assert isinstance(r, DecisionRow) and r.id == "r0" and set(r.questions) == {"dept", "urgent"}
        assert data.to_row(r) is r
        wire = r.to_wire()
        assert wire["id"] == "r0" and wire["questions"]["dept"]["type"] == "choice" and wire["weight"] == 1.0
        assert "id" not in DecisionRow(state="x", questions={"q": URGENT}).to_wire()

    def test_benchmark_rows(self) -> None:
        r = data.to_row({"id": "j1", "state": "Policy", "question": {"type": "noul"}, "expected": "yes"})
        assert r.labels == {"decision": True}
        assert data.from_benchmark({"state": "s", "question": ANGER, "expected": "2"}).labels == {"decision": 2}
        assert data.from_benchmark({"state": "s", "question": DEPT, "expected": "tech"}).labels == {"decision": "tech"}
        assert data.from_benchmark({"state": "s", "question": DEPT, "expected": None}).labels == {}
        assert data.from_benchmark({"state": "s", "question": URGENT, "expected": False}).labels == {"decision": False}

    def test_single_question_rows(self) -> None:
        soft = data.to_row({"state": "x", "question": DEPT, "target": [0.3, 0.7]})
        assert soft.labels == {"decision": [0.3, 0.7]}
        hard = data.to_row({"state": "x", "question": DEPT, "label": "tech", "weight": 2.0})
        assert hard.labels == {"decision": "tech"} and hard.weight == 2.0
        assert data.to_row({"state": "x", "question": URGENT, "target": 0.9}).labels == {"decision": 0.9}
        assert data.to_row({"state": "x", "question": URGENT, "target": True}).labels == {"decision": True}
        assert data.to_row({"state": "x", "question": URGENT}).labels == {}

    def test_rejects_unknown_shapes(self) -> None:
        with pytest.raises(ValueError):
            data.to_row({"state": "x"})
        with pytest.raises(TypeError):
            data.to_row("x")
        with pytest.raises(ValueError):
            DecisionRow(state="x", questions={})
        with pytest.raises(ValueError):
            DecisionRow(state="x", questions={"q": {"type": "nope"}})

    def test_load_and_save(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "rows.jsonl"
        n = data.save_rows(jsonl, rows(3))
        assert n == 3 and len(jsonl.read_text().splitlines()) == 3
        loaded = data.load_rows(jsonl)
        assert [r.id for r in loaded] == ["r0", "r1", "r2"] and loaded[0].labels["dept"] == "tech"
        as_json = tmp_path / "rows.json"
        as_json.write_text(json.dumps([r.to_wire() for r in loaded]))
        assert len(data.load_rows(as_json)) == 3
        as_json.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError):
            data.load_rows(as_json)


class TestLabels:
    def test_hard_labels(self) -> None:
        assert data.label_target(URGENT, True) == Target(label="true")
        assert data.label_target(URGENT, "no") == Target(label="false")
        assert data.label_target(DEPT, "billing") == Target(label="billing")
        assert data.label_target(ANGER, 2) == Target(label="2")
        assert data.label_target(ANGER, "2") == Target(label="2")  # the option id, as answers use it

    def test_soft_labels(self) -> None:
        assert data.label_target(URGENT, 0.75) == Target(probs={"false": 0.25, "true": 0.75})
        assert data.label_target(URGENT, 1).probs == {"false": 0.0, "true": 1.0}
        assert data.label_target(DEPT, {"billing": 0.7, "tech": 0.3}) == Target(probs={"billing": 0.7, "tech": 0.3})
        assert data.label_target(ANGER, [0.1, 0.6, 0.3]) == Target(probs={"0": 0.1, "1": 0.6, "2": 0.3})
        assert data.label_target(ANGER, {"0": 0.5, "1": 0.5, "2": 0.0}).probs == {"0": 0.5, "1": 0.5, "2": 0.0}

    @pytest.mark.parametrize(
        ("q", "label"),
        [
            (DEPT, "nope"),
            (DEPT, 1),
            (ANGER, 7),
            (ANGER, "7"),
            (ANGER, "high"),
            (URGENT, 2),
            (URGENT, "maybe"),
            (DEPT, {"billing": 0.7}),
            (DEPT, {"billing": 0.9, "tech": 0.9}),
            (ANGER, [0.5, 0.5]),
            (DEPT, {"billing": -0.5, "tech": 1.5}),
        ],
    )
    def test_bad_labels(self, q: object, label: object) -> None:
        with pytest.raises(ValueError):
            data.label_target(q, label)  # type: ignore[arg-type]

    def test_target_as_probs(self) -> None:
        assert Target(label="a").as_probs() == {"a": 1.0}
        assert Target(probs={"a": 0.5, "b": 0.5}).as_probs() == {"a": 0.5, "b": 0.5}


class TestDatums:
    def test_rows_to_datums(self) -> None:
        ds = data.rows_to_datums(rows(3))
        assert len(ds) == 6 and all(isinstance(d, Datum) for d in ds)
        assert ds[0].target == Target(label="tech") and ds[1].target == Target(label="false")
        wire = ds[0].to_wire()
        assert wire == {
            "state": {"body": "ticket 0"},
            "question": question_dict(DEPT),
            "target": {"label": "tech"},
            "weight": 1.0,
        }
        soft = data.rows_to_datums(
            [{"state": "x", "questions": {"d": DEPT}, "labels": {"d": {"billing": 0.6, "tech": 0.4}}}]
        )
        assert soft[0].to_wire()["target"] == {"probs": {"billing": 0.6, "tech": 0.4}}

    def test_unlabeled_and_unknown(self) -> None:
        assert data.rows_to_datums([{"state": "x", "questions": {"d": DEPT}}]) == []
        with pytest.raises(ValueError, match="unknown question"):
            data.rows_to_datums([{"state": "x", "questions": {"d": DEPT}, "labels": {"zz": "billing"}}])
        with pytest.raises(ValueError):
            Datum(state="x", question=DEPT, target=Target(label="a"), weight=-1)


class TestSplitAndBatches:
    def test_split(self) -> None:
        items = list(range(20))
        train, held = data.split(items, 0.1, seed=1)
        assert len(train) == 18 and len(held) == 2 and sorted(train + held) == items
        assert data.split(items, 0.1, seed=1) == (train, held)
        train0, held0 = data.split(items, 0.0)
        assert sorted(train0) == items and held0 == []
        assert data.split([], 0.5) == ([], [])
        assert len(data.split([1, 2, 3], 0.01)[1]) == 1  # at least one row when holdout > 0
        with pytest.raises(ValueError):
            data.split(items, 1.0)

    def test_batches(self) -> None:
        items = list(range(10))
        b = list(data.batches(items, 4, shuffle=False))
        assert b == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        shuffled = list(data.batches(items, 4, seed=1))
        assert sorted(x for batch in shuffled for x in batch) == items and shuffled != b
        assert list(data.batches(items, 4, seed=1)) == shuffled
        with pytest.raises(ValueError):
            list(data.batches(items, 0))
