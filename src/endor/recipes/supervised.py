"""Supervised fine-tuning with held-out evaluation: the default way to tune a decision model on your data.

    result = endor.recipes.supervised.train(SupervisedConfig(project="tickets", model_name="v1"), rows)
    client.system_one(state, questions, model=result.model)

What it does:

1. Splits off a held-out set (unless ``eval_rows`` is given) and expands rows into datums, one per labeled question.
2. Scores the untuned base model (``"<project>/base"``) on the held-out rows, so every later number has a baseline.
3. Trains one run: batches of ``batch_size``, a learning rate with linear warmup then linear decay,
   ``forward_backward`` and ``optim_step`` submitted together each step.
4. Every ``eval_every`` steps and at the end, scores the held-out datums with the run's current adapter and records
   accuracy, log loss, Brier, ECE and selective accuracy on the run's page.
5. Saves the final adapter as ``"<project>/<model_name>"`` and closes the run (also when anything fails).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .. import data as D
from .._headers import sdk_context
from ..metrics import decision_metrics
from ..types import Datum, LossFn, Target, answer_probabilities

if TYPE_CHECKING:
    from ..client import EndorClient
    from ..runs import Run

__all__ = ["SupervisedConfig", "SupervisedResult", "train", "evaluate_run", "evaluate_model", "lr_at"]


@dataclass
class SupervisedConfig:
    """Settings for ``train``."""

    project: str
    """The project that gets the run and the model (created with ``base_model`` if missing; a project without a
    base model gets ``base_model``)."""
    base_model: str = "decider-2b"
    """The base model the run trains on. With ``eval_base`` or replay rows it must be the project's base model,
    which they call as ``"<project>/base"``."""
    model_name: str | None = None
    """The saved model's name in the project; default ``"<name or sft>-<timestamp>"``."""
    rank: int = 16
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"
    """``"linear"``: warmup then linear decay to 0. ``"constant"``: warmup then flat."""
    warmup_steps: int = 10
    batch_size: int = 16
    epochs: int = 1
    loss_fn: LossFn = "cross_entropy"
    holdout: float = 0.1
    """Share of rows held out when ``eval_rows`` is not given."""
    eval_every: int = 0
    """Steps between held-out evaluations; 0 evaluates only at the end."""
    eval_base: bool = True
    """Score the untuned base model first."""
    seed: int = 0
    name: str | None = None
    """The run's name."""
    tags: list[str] = field(default_factory=list)
    replay_rows: Sequence[Any] | None = None
    """Optional unlabeled rows mixed into every batch with the base model's own answers as soft targets, which keeps
    the tuned model close to the base on everything outside your task. Off unless ``replay_per_batch`` > 0."""
    replay_per_batch: int = 0


@dataclass
class SupervisedResult:
    run_id: str
    model: str
    """``"<project>/<model_name>"``."""
    base_metrics: dict[str, Any] | None
    """Held-out metrics of the untuned base, or None when ``eval_base`` was off."""
    history: list[dict[str, Any]] = field(default_factory=list)
    """``[{"step": int, "metrics": {...}}]`` for every held-out evaluation during and after training."""

    @property
    def final_metrics(self) -> dict[str, Any]:
        return dict(self.history[-1]["metrics"]) if self.history else {}


def lr_at(cfg: SupervisedConfig, step: int, total: int) -> float:
    """The learning rate at ``step`` of ``total``: linear warmup, then linear decay or constant."""
    warm = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
    decay = (1 - step / total) if cfg.lr_schedule == "linear" else 1.0
    return cfg.learning_rate * warm * decay


def evaluate_run(run: Run, datums: Sequence[Datum]) -> dict[str, Any]:
    """Held-out metrics for a run's current adapter (``run.forward``; no model is saved)."""
    out = run.forward(datums).result()
    return decision_metrics(out.probabilities, [d.target for d in datums])


def evaluate_model(client: EndorClient, model: str, rows: Sequence[Any]) -> dict[str, Any]:
    """Held-out metrics for any model through ``system_one``: one request per row with all its labeled questions."""
    probs, targets = [], []
    for row in map(D.to_row, rows):
        asked = {n: row.questions[n] for n in row.labels}
        if not asked:
            continue
        res = client.system_one(row.state, asked, model=model)
        for n, q in asked.items():
            probs.append(answer_probabilities(res.answers[n]))
            targets.append(D.label_target(q, row.labels[n]))
    return decision_metrics(probs, targets)


def _replay_datums(client: EndorClient, cfg: SupervisedConfig, base: str) -> list[Datum]:
    if not cfg.replay_rows or cfg.replay_per_batch <= 0:
        return []
    out = []
    for row in map(D.to_row, cfg.replay_rows):
        res = client.system_one(row.state, row.questions, model=base)
        out += [
            Datum(state=row.state, question=q, target=Target(probs=answer_probabilities(res.answers[n])))
            for n, q in row.questions.items()
        ]
    return out


def train(
    cfg: SupervisedConfig,
    rows: Sequence[Any],
    eval_rows: Sequence[Any] | None = None,
    client: EndorClient | None = None,
) -> SupervisedResult:
    """Fine-tune on ``rows`` and return the saved model id with its held-out metrics.

    ``rows`` and ``eval_rows`` are decision rows in any supported shape. Without ``eval_rows``, ``cfg.holdout`` of
    ``rows`` is held out. Pass your own ``client`` to reuse it; otherwise one is created from the environment.
    """
    from ..client import EndorClient  # late import: the recipes import without an API key

    with sdk_context(recipe="supervised"):
        rows = [D.to_row(r) for r in rows]
        if eval_rows is None:
            rows, eval_rows = D.split(rows, cfg.holdout, cfg.seed)
        eval_rows = [D.to_row(r) for r in eval_rows]
        train_datums, eval_datums = D.rows_to_datums(rows), D.rows_to_datums(eval_rows)
        if not train_datums:
            raise ValueError("no labeled questions to train on")
        own = client is None
        client = client or EndorClient()
        try:
            project = client.projects.get_or_create(cfg.project, base_model=cfg.base_model)
            base = f"{project.name}/base"  # decisions name a project: the untuned base is "<project>/base"
            uses_base = (cfg.eval_base and eval_datums) or (cfg.replay_rows and cfg.replay_per_batch > 0)
            if uses_base and project.info_.base_model is None:
                project.update(base_model=cfg.base_model)  # what the first run would set anyway
            elif uses_base and project.info_.base_model != cfg.base_model:
                raise ValueError(
                    f"project {project.name!r} has base model {project.info_.base_model!r}, not "
                    f"{cfg.base_model!r}: {base} would score the wrong base. Use the project's base model, another "
                    "project, or eval_base=False without replay rows"
                )
            base_metrics = evaluate_model(client, base, eval_rows) if cfg.eval_base and eval_datums else None
            replay = _replay_datums(client, cfg, base)
            with project.runs.create(
                cfg.base_model,
                rank=cfg.rank,
                seed=cfg.seed,
                name=cfg.name,
                tags=cfg.tags,
                config={k: v for k, v in asdict(cfg).items() if k != "replay_rows"},
            ) as run:  # closed on success and on any error, so its GPU is released
                if base_metrics is not None:
                    run.log_eval(base, base_metrics, step=0, name="heldout")
                total = math.ceil(len(train_datums) / cfg.batch_size) * cfg.epochs
                result = SupervisedResult(run.id, "", base_metrics)

                def held_out(step: int) -> None:
                    if eval_datums:
                        metrics = evaluate_run(run, eval_datums)
                        run.log_eval(f"{run.id}@{step}", metrics, step=step, name="heldout")
                        result.history.append({"step": step, "metrics": metrics})

                step = 0
                for epoch in range(cfg.epochs):
                    for batch in D.batches(train_datums, cfg.batch_size, seed=cfg.seed + epoch):
                        if replay:
                            k = cfg.replay_per_batch
                            batch = batch + [replay[(step * k + i) % len(replay)] for i in range(k)]
                        fb = run.forward_backward(batch, cfg.loss_fn)
                        opt = run.optim_step(learning_rate=lr_at(cfg, step, total))
                        fb.result()
                        opt.result()
                        step += 1
                        if cfg.eval_every and step % cfg.eval_every == 0 and step < total:
                            held_out(step)
                held_out(step)
                name = cfg.model_name or f"{cfg.name or 'sft'}-{datetime.now():%Y%m%d-%H%M%S}"
                result.model = run.save_checkpoint(name).result()
            return result
        finally:
            if own:
                client.close()
