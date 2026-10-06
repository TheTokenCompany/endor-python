"""Teacher distillation: label unlabeled rows with a teacher of your choice, then fine-tune on the teacher's
answer distributions with the supervised recipe.

    class MyTeacher:
        def label(self, rows):                       # -> one {question name: label} per row
            return [{"dept": {"billing": 0.7, "technical": 0.3}} for _ in rows]

    cfg = DistillConfig(SupervisedConfig(project="tickets"), teacher=MyTeacher())
    result = distill(cfg, unlabeled_rows, eval_rows)  # eval_rows: rows with gold labels
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .. import data as D
from .._headers import sdk_context
from ..types import DecisionRow
from .supervised import SupervisedConfig, SupervisedResult, train

if TYPE_CHECKING:
    from ..client import EndorClient

__all__ = ["Teacher", "DistillConfig", "label_with_teacher", "distill"]


class Teacher(Protocol):
    """Anything that labels decision rows.

    ``label`` returns one ``{question name: label}`` per row, in order. A label may be hard (an option key, a bool,
    a level index) or soft (a distribution over the options, see docs/DATA_FORMAT.md). Return None for a question
    the teacher cannot answer; it stays unlabeled.
    """

    def label(self, rows: Sequence[DecisionRow]) -> Sequence[dict[str, Any] | None]: ...


@dataclass
class DistillConfig:
    supervised: SupervisedConfig
    """Where and how the labeled rows are trained."""
    teacher: Teacher
    budget: float = 1.0
    """Share of the unlabeled rows sent to the teacher (chosen at random with ``seed``)."""
    seed: int = 0


def label_with_teacher(rows: Sequence[Any], cfg: DistillConfig) -> list[DecisionRow]:
    """Teacher labels for the unlabeled questions of a random ``cfg.budget`` share of ``rows``.
    Labels the rows already have are kept."""
    if not 0 < cfg.budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    parsed = [D.to_row(r) for r in rows]
    rng = random.Random(cfg.seed)
    chosen = [r for r in parsed if rng.random() < cfg.budget] if cfg.budget < 1 else parsed
    out = []
    for row, labels in zip(chosen, cfg.teacher.label(chosen), strict=False):
        new = {k: v for k, v in (labels or {}).items() if v is not None and k in row.questions}
        merged = {**new, **row.labels}
        out.append(row.model_copy(update={"labels": merged}))
    return out


def distill(
    cfg: DistillConfig,
    unlabeled_rows: Sequence[Any],
    eval_rows: Sequence[Any],
    client: EndorClient | None = None,
) -> SupervisedResult:
    """Label with the teacher, then fine-tune with the supervised recipe; evaluated on ``eval_rows``."""
    with sdk_context(recipe="distill"):
        return train(cfg.supervised, label_with_teacher(unlabeled_rows, cfg), eval_rows, client)
