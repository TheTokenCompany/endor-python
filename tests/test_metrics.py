from __future__ import annotations

import math

import pytest

from endor import Target
from endor.metrics import decision_metrics


def test_perfect_predictions() -> None:
    m = decision_metrics([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], [Target(label="a"), Target(label="b")])
    assert m["n"] == 2 and m["accuracy"] == 1.0 and m["brier"] == 0.0 and m["ece"] == 0.0
    assert math.isclose(m["nll"], 0.0, abs_tol=1e-9) and m["mean_confidence"] == 1.0
    assert m["selective"]["0.95"] == {"coverage": 1.0, "accuracy": 1.0}


def test_mixed_predictions() -> None:
    probs = [{"a": 0.9, "b": 0.1}, {"a": 0.2, "b": 0.8}, {"a": 0.6, "b": 0.4}]
    targets = [Target(label="a"), Target(label="b"), Target(label="b")]
    m = decision_metrics(probs, targets)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["nll"] == pytest.approx((-math.log(0.9) - math.log(0.8) - math.log(0.4)) / 3)
    assert m["brier"] == pytest.approx(((0.01 + 0.01) + (0.04 + 0.04) + (0.36 + 0.36)) / 3)
    assert m["mean_confidence"] == pytest.approx((0.9 + 0.8 + 0.6) / 3)
    assert m["selective"]["0.7"] == {"coverage": pytest.approx(2 / 3), "accuracy": 1.0}
    assert m["selective"]["0.95"] == {"coverage": 0.0, "accuracy": None}
    # bins (0.8, 0.9]: conf 0.9 and 0.8, both correct -> |0.85 - 1| * 2/3 ; bin (0.5, 0.6]: conf 0.6, wrong -> 0.6 * 1/3
    assert m["ece"] == pytest.approx(abs(0.85 - 1.0) * 2 / 3 + 0.6 / 3)


def test_soft_targets_use_argmax_for_accuracy() -> None:
    m = decision_metrics([{"a": 0.6, "b": 0.4}], [Target(probs={"a": 0.3, "b": 0.7})])
    assert m["accuracy"] == 0.0
    assert m["nll"] == pytest.approx(-(0.3 * math.log(0.6) + 0.7 * math.log(0.4)))


def test_empty_and_mismatch() -> None:
    assert decision_metrics([], []) == {"n": 0}
    with pytest.raises(ValueError):
        decision_metrics([{"a": 1.0}], [])


def test_zero_probability_is_clamped() -> None:
    m = decision_metrics([{"a": 0.0, "b": 1.0}], [Target(label="a")])
    assert math.isfinite(m["nll"]) and m["nll"] > 20
