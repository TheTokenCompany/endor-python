"""Decision rows: loading, format conversion, labels to training datums, splits and batches.

A row is a decision request with labels (README, "Data format"):

    {"id"?, "state", "questions": {name: question}, "labels"?: {name: label}, "weight"?}

Two other row shapes are converted automatically, each becoming one question named ``"decision"``:

    {"state", "question", "expected"}          benchmark rows: ``expected`` is "yes"/"no", an option key or a level
    {"state", "question", "label"? , "target"?} single-question rows: ``target`` may be a soft distribution
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

from .types import Datum, DecisionRow, Label, Question, Target, option_keys, question_dict

__all__ = [
    "to_row",
    "from_benchmark",
    "from_single",
    "load_rows",
    "save_rows",
    "label_target",
    "row_to_datums",
    "rows_to_datums",
    "split",
    "batches",
]

T = TypeVar("T")


def to_row(obj: Any) -> DecisionRow:
    """Any supported row shape (dict or ``DecisionRow``) as a ``DecisionRow``."""
    if isinstance(obj, DecisionRow):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"a row is a dict or a DecisionRow, got {type(obj).__name__}")
    if "questions" in obj:
        return DecisionRow.model_validate(obj)
    if "question" in obj and "expected" in obj:
        return from_benchmark(obj)
    if "question" in obj:
        return from_single(obj)
    raise ValueError("unrecognized row: expected a 'questions' map, or a single 'question'")


def from_benchmark(r: dict[str, Any]) -> DecisionRow:
    """A ``{"state", "question", "expected"}`` row as a ``DecisionRow`` with one question, ``"decision"``."""
    q, exp = r["question"], r.get("expected")
    qtype = question_dict(q)["type"]
    labels: dict[str, Any]
    if exp is None:
        labels = {}
    elif qtype == "noul":
        labels = {"decision": exp if isinstance(exp, bool) else str(exp).lower() in ("yes", "true", "1")}
    elif qtype == "score":
        labels = {"decision": int(exp)}
    else:
        labels = {"decision": str(exp)}
    return DecisionRow(
        id=r.get("id"),
        state=r["state"],
        questions={"decision": q},
        labels=labels,
        weight=r.get("weight", 1.0),
    )


def from_single(r: dict[str, Any]) -> DecisionRow:
    """A ``{"state", "question", "label"?, "target"?}`` row as a ``DecisionRow`` with one question, ``"decision"``.
    A list or float ``target`` is a soft label (per-option probabilities, or P(true) for a noul)."""
    q = r["question"]
    tgt, lab = r.get("target"), r.get("label")
    label: Any
    if isinstance(tgt, (list, float, dict)) and not isinstance(tgt, bool):
        label = tgt
    elif lab is not None:
        label = lab
    else:
        label = tgt
    return DecisionRow(
        id=r.get("id"),
        state=r["state"],
        questions={"decision": q},
        labels={} if label is None else {"decision": label},
        weight=r.get("weight", 1.0),
    )


def load_rows(path: str | Path) -> list[DecisionRow]:
    """Rows from a ``.jsonl`` file (one row per line) or a ``.json`` file (an array), in any supported shape."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError(f"{p}: a .json file must hold an array of rows")
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [to_row(x) for x in items]


def save_rows(path: str | Path, rows: Iterable[Any]) -> int:
    """Write rows as ``.jsonl`` in the native shape. Returns the number of rows written."""
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(to_row(r).to_wire(), ensure_ascii=False) + "\n")
            n += 1
    return n


def label_target(question: Question, label: Label) -> Target:
    """A row label as a ``Target`` over ``option_keys(question)``.

    noul: ``True``/``False``, or P(true) as a float, or ``{"true": p, "false": 1-p}``.
    choice: an option key, or ``{key: p, ...}`` over every option.
    score: a level index (``2``) or its option id (``"2"``), or ``[p0, p1, ...]`` or ``{"0": p0, ...}``.
    """
    keys, qtype = option_keys(question), question_dict(question)["type"]
    if isinstance(label, dict):
        probs = {str(k): float(v) for k, v in label.items()}
        _check_probs(probs, keys)
        return Target(probs=probs)
    if isinstance(label, list):
        if len(label) != len(keys):
            raise ValueError(f"a soft label needs {len(keys)} probabilities, got {len(label)}")
        probs = dict(zip(keys, map(float, label), strict=False))
        _check_probs(probs, keys)
        return Target(probs=probs)
    if qtype == "noul":
        if isinstance(label, bool):
            return Target(label="true" if label else "false")
        if isinstance(label, (int, float)) and 0 <= label <= 1:
            return Target(probs={"false": 1 - float(label), "true": float(label)})
        if isinstance(label, str) and label.lower() in ("true", "false", "yes", "no"):
            return Target(label="true" if label.lower() in ("true", "yes") else "false")
    if qtype == "score" and (
        (isinstance(label, int) and not isinstance(label, bool)) or (isinstance(label, str) and label.isdigit())
    ):
        if str(label) not in keys:
            raise ValueError(f"score level {label!r} is not an option of this question (0 to {len(keys) - 1})")
        return Target(label=str(label))
    if qtype == "choice" and isinstance(label, str):
        if label not in keys:
            raise ValueError(f"label {label!r} is not an option ({keys})")
        return Target(label=label)
    raise ValueError(f"label {label!r} does not fit a {qtype} question")


def _check_probs(probs: dict[str, float], keys: list[str]) -> None:
    if set(probs) != set(keys):
        raise ValueError(f"a soft label must cover exactly the options {keys}, got {sorted(probs)}")
    if min(probs.values()) < 0 or not math.isclose(sum(probs.values()), 1.0, abs_tol=0.02):
        raise ValueError("a soft label must be non-negative and sum to 1")


def row_to_datums(row: Any) -> list[Datum]:
    """One ``Datum`` per labeled question of a row; unlabeled questions are skipped."""
    r = to_row(row)
    out = []
    for name, label in r.labels.items():
        if name not in r.questions:
            raise ValueError(f"label for unknown question {name!r}")
        q = r.questions[name]
        out.append(Datum(state=r.state, question=q, target=label_target(q, label), weight=r.weight))
    return out


def rows_to_datums(rows: Iterable[Any]) -> list[Datum]:
    """One ``Datum`` per labeled question across all rows."""
    return [d for r in rows for d in row_to_datums(r)]


def split(rows: Sequence[T], holdout: float = 0.1, seed: int = 0) -> tuple[list[T], list[T]]:
    """``(train, held_out)``: a seeded shuffle, then the last ``holdout`` share held out (at least one row)."""
    if not 0 <= holdout < 1:
        raise ValueError("holdout must be in [0, 1)")
    items = list(rows)
    random.Random(seed).shuffle(items)
    n_out = max(1, math.floor(len(items) * holdout)) if holdout > 0 and items else 0
    return items[: len(items) - n_out], items[len(items) - n_out :]


def batches(items: Sequence[T], size: int, *, shuffle: bool = True, seed: int = 0) -> Iterator[list[T]]:
    """Yield lists of ``size`` items (the last one may be shorter), in a seeded shuffled order."""
    if size < 1:
        raise ValueError("size must be >= 1")
    order = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), size):
        yield [items[j] for j in order[i : i + size]]
