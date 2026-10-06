"""Decision metrics on held-out data: accuracy, log loss, Brier score, calibration (ECE) and selective accuracy.

    out = run.forward(datums).result()
    decision_metrics(out.probabilities, [d.target for d in datums])

Inputs are per-decision option distributions (``{option_id: p}``) and their ``Target`` objects. A soft target counts
its most likely option as the correct answer for accuracy. The same definitions are used by server-side evaluations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .types import Target

__all__ = ["decision_metrics", "THRESHOLDS"]

THRESHOLDS = (0.5, 0.7, 0.9, 0.95)
"""Confidence thresholds for selective accuracy."""


def decision_metrics(probs: Sequence[dict[str, float]], targets: Sequence[Target], bins: int = 10) -> dict[str, Any]:
    """Metrics over paired distributions and targets.

    Returns ``{"n", "accuracy", "nll", "brier", "ece", "mean_confidence", "selective": {"0.5": {"coverage",
    "accuracy"}, ...}}``; ``{"n": 0}`` for empty input.

    - accuracy: argmax(p) equals argmax(y)
    - nll: −Σ y log p, averaged
    - brier: Σ (p − y)², averaged
    - ece: expected calibration error with ``bins`` equal-width bins on top-1 confidence
    - selective: coverage and accuracy among decisions whose top-1 confidence is at least the threshold
    """
    if len(probs) != len(targets):
        raise ValueError("probs and targets must have the same length")
    n = len(probs)
    if n == 0:
        return {"n": 0}
    acc = nll = brier = 0.0
    conf: list[float] = []
    correct: list[float] = []
    for p, t in zip(probs, targets, strict=False):
        y = t.as_probs()
        pred = max(p, key=lambda k: p[k])
        gold = max(y, key=lambda k: y[k])
        ok = float(pred == gold)
        acc += ok
        nll += -sum(w * math.log(max(p.get(k, 0.0), 1e-12)) for k, w in y.items() if w > 0)
        brier += sum((p.get(k, 0.0) - y.get(k, 0.0)) ** 2 for k in set(p) | set(y))
        conf.append(p[pred])
        correct.append(ok)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            avg_conf = sum(conf[i] for i in idx) / len(idx)
            avg_acc = sum(correct[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(avg_conf - avg_acc)
    selective: dict[str, dict[str, float | None]] = {}
    for th in THRESHOLDS:
        idx = [i for i, c in enumerate(conf) if c >= th]
        selective[str(th)] = {
            "coverage": len(idx) / n,
            "accuracy": (sum(correct[i] for i in idx) / len(idx)) if idx else None,
        }
    return {
        "n": n,
        "accuracy": acc / n,
        "nll": nll / n,
        "brier": brier / n,
        "ece": ece,
        "mean_confidence": sum(conf) / n,
        "selective": selective,
    }
