"""Public data types.

Decisions
    Noul, Choice, Score          the three question types (plain dicts in the same shape are accepted too)
    NoulAnswer, ChoiceAnswer, ScoreAnswer, SystemOneResponse, Usage
    ModelMetadata, ListModelsResponse, BaseModelInfo

Training
    Datum        one state + one question + its Target (+ weight): the unit of ``forward`` / ``forward_backward``
    DecisionRow  one state + named questions + labels: the unit of datasets (README, "Data format")
    Target, LoraConfig, AdamParams, ForwardOutput, OptimStepOutput

Resources (read-only views of what the API returns)
    ProjectInfo, RunInfo, ModelInfo, DatasetInfo, Evaluation, MetricPoint, DownloadedFile, UsageRow, WhoAmI, FutureInfo

Option ids are the only identifiers used for a question's options anywhere in the SDK:
    noul   -> "false", "true"
    choice -> the criteria keys, in the order given
    score  -> "0" .. "n-1" (the level index)
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer

__all__ = [
    "JSONContent",
    "Question",
    "QuestionType",
    "LossFn",
    "Label",
    "Noul",
    "Choice",
    "Score",
    "NoulCriteria",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "Answer",
    "Usage",
    "SystemOneResponse",
    "ModelMetadata",
    "ListModelsResponse",
    "BaseModelInfo",
    "Target",
    "Datum",
    "DecisionRow",
    "LoraConfig",
    "LoraInfo",
    "AdamParams",
    "ForwardOutput",
    "OptimStepOutput",
    "ProjectInfo",
    "WandbSettings",
    "RunInfo",
    "ModelInfo",
    "DatasetInfo",
    "Evaluation",
    "MetricPoint",
    "DownloadedFile",
    "UsageRow",
    "WhoAmI",
    "FutureInfo",
    "question_dict",
    "option_keys",
    "answer_probabilities",
]

JSONContent = str | dict[str, Any] | list[Any]
"""Text or JSON: the type of a state, of instructions, of option descriptions and of score levels."""
QuestionType = Literal["noul", "choice", "score"]
LossFn = Literal["cross_entropy", "brier"]
Label = bool | int | str | float | dict[str, float] | list[float]
"""A row label in the natural form for its question type (README, "Data format")."""

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


# ---------------------------------------------------------------- questions


class _Question(BaseModel):
    """Rejects unknown fields; omits unset optional fields from the wire form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {k: v for k, v in handler(self).items() if v is not None}


class NoulCriteria(BaseModel):
    """Optional descriptions of the yes and no outcomes of a Noul."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    true: JSONContent | None = None
    false: JSONContent | None = None


class Noul(_Question):
    """A yes/no question. The answer is ``P(true)``.

    ```python
    Noul(instructions="Does the customer need an answer today?")
    ```
    """

    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    """What to decide, as text or JSON. Refer to parts of the state with backticked paths, as in
    ``"Is `ticket.body` angry?"``."""
    criteria: NoulCriteria | None = None
    """Optional descriptions of the ``true`` and ``false`` outcomes."""


class Choice(_Question):
    """One of several named options. The answer is a probability for every option.

    ```python
    Choice(instructions="Route the ticket", criteria={"billing": "Payments, refunds", "technical": None})
    ```
    """

    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]
    """Option key → description (or None). The order of the keys is the order of the options. 1 to 255 options;
    some base models read fewer (see ``EndorClient.base_models``)."""

    @field_validator("criteria")
    @classmethod
    def _n_options(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not 1 <= len(v) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"a choice has 1 to {MAX_CHOICE_OPTIONS} options, got {len(v)}")
        return v


class Score(_Question):
    """One of several ordered levels. The answer is the expected level plus a probability per level.

    ``criteria`` are the levels of one scale, lowest first: each item describes one level, not a thing to check.
    The answer's ``score`` is the expected level, from 0 (the first item) to n - 1 (the last).

    ```python
    Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"])
    Score(instructions="How Finnish is this name?",
          criteria=["Clearly not Finnish", "Possibly Finnish", "Unmistakably Finnish"])
    ```
    """

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: list[JSONContent]
    """The levels of one scale, lowest first (not a list of things to check). 2 to 10 levels; level ``i`` has option
    id ``str(i)``."""

    @field_validator("criteria")
    @classmethod
    def _n_levels(cls, v: list[Any]) -> list[Any]:
        if not MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS:
            raise ValueError(f"a score has {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {len(v)}")
        return v


Question = Noul | Choice | Score | Mapping[str, Any]
"""A question object, or a plain dict in the same shape (``{"type": "noul", "instructions": ...}``)."""


def question_dict(q: Question) -> dict[str, Any]:
    """The wire form of a question."""
    if isinstance(q, BaseModel):
        return q.model_dump(mode="json")
    if isinstance(q, Mapping):
        d = dict(q)
        if not isinstance(d.get("type"), str) or d["type"] not in ("noul", "choice", "score"):
            raise ValueError(f"a question dict needs a type of noul, choice or score, got {d.get('type')!r}")
        if d["type"] in ("choice", "score") and "criteria" not in d:
            raise ValueError(f"a {d['type']} question needs criteria")
        return d
    raise TypeError(f"expected a Noul, Choice, Score or a question dict, got {type(q).__name__}")


def option_keys(q: Question) -> list[str]:
    """The canonical option ids of a question: ``["false", "true"]``, the criteria keys, or ``["0", ..., "n-1"]``."""
    d = question_dict(q)
    if d["type"] == "noul":
        return ["false", "true"]
    if d["type"] == "choice":
        return [str(k) for k in d["criteria"]]
    return [str(i) for i in range(len(d["criteria"]))]


# ---------------------------------------------------------------- answers


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class NoulAnswer(_Answer):
    """The answer to a Noul."""

    type: Literal["noul"] = "noul"
    noul: float
    """P(true)."""

    @property
    def probabilities(self) -> dict[str, float]:
        """``{"false": 1 - noul, "true": noul}``."""
        return {"false": 1 - self.noul, "true": self.noul}


class ChoiceAnswer(_Answer):
    """The answer to a Choice."""

    type: Literal["choice"] = "choice"
    choice: str
    """The most likely option."""
    probabilities: dict[str, float]
    """A probability for every option, by option key."""
    confidence: float
    """How far the top probability is above uniform: ``(p_max - 1/n) / (1 - 1/n)``, in [0, 1]."""


class ScoreAnswer(_Answer):
    """The answer to a Score."""

    type: Literal["score"] = "score"
    score: float
    """The expected level, ``Σ i · p_i``."""
    legend: dict[str, JSONContent | None]
    """Level index (as a string) → the level's description."""
    probabilities: dict[str, float]
    """Level index (as a string) → probability."""
    confidence: float
    """How concentrated the distribution is around its mode, in [0, 1]."""


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
"""An answer, identified by its ``type``."""


class Usage(_Answer):
    """Token usage of a decision request."""

    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(BaseModel):
    """The answers to a decision request, by question name.

    Subclass it to get typed attributes per question (``response_model=``):

    ```python
    class Routed(SystemOneResponse):
        dept: ChoiceAnswer
        urgent: NoulAnswer

    res = client.system_one(state, questions, response_model=Routed)
    res.dept.choice
    ```
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    """The model that answered: ``"<project>/<base id>"`` or ``"<project>/<name>"``."""
    answers: dict[str, Answer] = Field(default_factory=dict)
    """Every answer, by question name."""
    usage: Usage = Field(default_factory=Usage)

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}


def answer_probabilities(answer: Answer) -> dict[str, float]:
    """An answer as a distribution over ``option_keys``: noul → false/true, score → ``"0"``..``"n-1"``."""
    if isinstance(answer, NoulAnswer):
        return answer.probabilities
    return {str(k): float(v) for k, v in answer.probabilities.items()}


class BaseModelInfo(BaseModel):
    """A base decision model you can fine-tune and decide with, through a project that has it
    (``"<project>/<id>"``; add it with ``project.add_base_model(id)``)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    description: str | None = None
    release_date: str | None = None
    params: str | None = None
    """The model size, for example ``"26B (4B active)"``."""
    provider: str | None = None
    """Who published it."""
    leaderboard_rank: int | None = None
    """Its rank on the Jev Decision Index, when listed."""
    max_options: int | None = None
    """The most options a Choice (or levels a Score) may have on this base."""
    max_question_tokens: int | None = None
    """The most tokens one rendered question (state included) may have."""
    trainable: bool = True
    default_rank: int = 16
    max_rank: int | None = None
    lora_targets: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    """What a decision's state may hold: ``"text"``, and ``"image"`` on bases that read images (``endor.Image``)."""
    contract: str | None = None
    """How a decision becomes the model's input and which outputs give the probabilities."""
    contract_version: int | None = None
    hf_repo: str | None = None
    """The Hugging Face repository of the weights; a downloaded adapter is a PEFT LoRA for it."""
    hf_revision: str | None = None
    """The pinned commit of ``hf_repo`` that Endor loads."""
    trainer_gpu: str | None = None
    price_per_mtok_decide: float | None = None
    """USD per 1M decision input tokens on this base (None while unpriced)."""
    price_per_mtok_decide_continuous_learning: float | None = None
    """The same in a managed project (Endor trains it from its decisions): always this price, paused or not."""
    price_per_gpu_hour: float | None = None
    """USD per training GPU-hour, from when a run's GPU is requested until it is released (None while unpriced)."""


class ModelMetadata(BaseModel):
    """One entry of ``client.models.list()``: a name you can pass as ``model``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str
    """The id to pass as ``model``."""
    description: str = ""
    release_date: str | None = None
    endor: dict[str, Any] = Field(default_factory=dict)
    """Endor's metadata: ``kind``, ``project`` and ``base_model``; for ``project`` also ``model`` (what ``<project>``
    calls now), for ``base`` ``contract``, and for saved models ``training_run_id``, ``step`` and
    ``parent_model``."""

    @property
    def kind(self) -> str:
        """``"project"`` (``<project>``), ``"base"`` (``<project>/<base id>``) or ``"model"``
        (``<project>/<name>``)."""
        if self.endor.get("kind"):
            return str(self.endor["kind"])
        return "project" if "/" not in self.name else "model"


class ListModelsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    models: list[ModelMetadata]


# ---------------------------------------------------------------- training


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Target(_Strict):
    """What the loss compares against: a hard ``label`` (an option id) or soft ``probs`` over every option."""

    label: str | None = None
    probs: dict[str, float] | None = None

    def as_probs(self) -> dict[str, float]:
        """The target as a distribution (one-hot for a label)."""
        if self.probs is not None:
            return self.probs
        return {str(self.label): 1.0}


class Datum(_Strict):
    """One training example: a state, one question, its target and a weight."""

    state: JSONContent
    question: Any
    """A Noul, Choice, Score or question dict."""
    target: Target
    weight: float = Field(1.0, ge=0)

    def to_wire(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "question": question_dict(self.question),
            "target": self.target.model_dump(exclude_none=True),
            "weight": self.weight,
        }


class DecisionRow(_Strict):
    """One dataset row: a decision request plus labels (README, "Data format")."""

    id: str | None = None
    state: JSONContent
    questions: dict[str, Any]
    """Question name → Noul, Choice, Score or question dict."""
    labels: dict[str, Any] = Field(default_factory=dict)
    """Question name → label. Questions without a label are unlabeled."""
    weight: float = Field(1.0, ge=0)

    @field_validator("questions")
    @classmethod
    def _some_questions(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("a row needs at least one question")
        for q in v.values():
            question_dict(q)
        return v

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "state": self.state,
            "questions": {k: question_dict(q) for k, q in self.questions.items()},
            "labels": self.labels,
            "weight": self.weight,
        }
        if self.id is not None:
            out["id"] = self.id
        return out


class LoraConfig(_Strict):
    """How a run's adapter is shaped."""

    rank: int = Field(16, ge=1, le=256)
    alpha: float = 32.0
    seed: int | None = None
    train_attn: bool = True
    train_mlp: bool = True
    train_readout: bool = False


class AdamParams(_Strict):
    """AdamW hyperparameters for one ``optim_step``. You own the learning-rate schedule."""

    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
    """0 disables clipping."""


# ---------------------------------------------------------------- read-only views


class _View(BaseModel):
    """Read-only views of API responses: unknown fields are ignored, so newer servers don't break older SDKs."""

    model_config = ConfigDict(extra="ignore")


class LoraInfo(LoraConfig):
    """A run's adapter settings as the API reports them (unknown fields ignored)."""

    model_config = ConfigDict(extra="ignore")


class ForwardOutput(_View):
    """The result of ``forward`` or ``forward_backward``."""

    outputs: list[dict[str, Any]]
    """Per datum, in input order: ``{"probabilities": {option_id: p}, "loss": float}``."""
    metrics: dict[str, float]
    """``loss:sum``, ``weight:sum``, ``loss:mean``, ``accuracy``, ``n``."""

    @property
    def probabilities(self) -> list[dict[str, float]]:
        """Per datum, in input order: every option's probability."""
        return [dict(o["probabilities"]) for o in self.outputs]

    @property
    def losses(self) -> list[float]:
        """Per datum, in input order: its loss."""
        return [float(o["loss"]) for o in self.outputs]

    @property
    def loss(self) -> float:
        """The weighted mean loss over the batch."""
        return self.metrics.get("loss:mean", float("nan"))

    @property
    def accuracy(self) -> float:
        """The share of datums whose most likely option is the target's."""
        return self.metrics.get("accuracy", float("nan"))


class OptimStepOutput(_View):
    """The result of ``optim_step``: the new step, the gradient norm before clipping and the learning rate used."""

    step: int
    grad_norm: float
    learning_rate: float


class WandbSettings(_View):
    """A project's Weights & Biases logging, set in its Settings tab or with ``project.update(wandb=...)``. Endor logs
    runs from its servers through the organization's W&B connection (dashboard: Settings > Integrations), to the W&B
    project ``endor-<project name>`` in the connection's entity."""

    enabled: bool = False
    """New runs log to W&B (``runs.create(wandb=...)`` overrides it per run)."""
    url: str | None = None
    """That W&B project's page, while logging is on and the organization is connected."""


class ProjectInfo(_View):
    name: str
    description: str | None = None
    kind: str = "custom"
    """``custom`` (you train models with the SDK) or ``managed`` (Endor trains new versions from the project's
    decisions). Set at creation; it never changes."""
    base_models: list[str] = Field(default_factory=list)
    """The project's base model ids, in the order they were added. Each answers as ``"<project>/<base id>"``, and the
    first also as ``"<project>"`` (in a managed project, until its first version). Change them with
    ``project.add_base_model`` and ``project.remove_base_model``."""
    paused: bool | None = None
    """Managed projects: learning is paused (the project keeps serving its newest version). None for custom."""
    n_datasets: int = 0
    n_runs: int = 0
    n_models: int = 0
    """Saved models plus base models."""
    wandb: WandbSettings | None = None
    """Weights & Biases logging of the project's runs."""
    created_at: datetime


class RunInfo(_View):
    id: str
    project: str
    name: str | None = None
    base_model: str
    contract: str | None = None
    lora: LoraInfo = Field(default_factory=LoraInfo)
    status: str
    """``provisioning``, ``ready``, ``idle`` (parked after 2 minutes without calls; the next call resumes it),
    ``closing`` (closed, finishing accepted calls), ``closed`` or ``failed``. Treat unknown values as active."""
    ready_future_id: str | None = None
    step: int = 0
    """Optimizer steps taken."""
    total_steps: int | None = None
    """The optimizer steps planned (``runs.create(total_steps=...)`` or ``run.set_total_steps``); None if unknown."""
    seconds_per_step: float | None = None
    """The median time per optimizer step over the last 20 steps; None before the second step."""
    next_seq_id: int = 0
    parent_model: str | None = None
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    code_hash: str | None = None
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    failure: dict[str, Any] | None = None
    """``{"code", "message"}`` once the run failed (e.g. ``trainer_lost``)."""
    created_at: datetime
    ready_at: datetime | None = None
    """When the run's first GPU was ready (``created_at`` to ``ready_at``: waiting for a GPU)."""
    closed_at: datetime | None = None
    wandb: bool = False
    """The run logs to Weights & Biases."""
    wandb_url: str | None = None
    """Its W&B run, once Endor has created it (within about a minute of the first metrics)."""

    @property
    def progress(self) -> float | None:
        """``step / total_steps`` (0 to 1), or None without a total."""
        if not self.total_steps:
            return None
        return min(1.0, self.step / self.total_steps)

    @property
    def eta_seconds(self) -> float | None:
        """Seconds until ``total_steps`` at the recent pace, or None without a total or a pace."""
        if not self.total_steps or self.seconds_per_step is None:
            return None
        return max(0, self.total_steps - self.step) * self.seconds_per_step


class ModelInfo(_View):
    id: str
    """``"<project>/<name>"`` or ``"<project>/<base id>"``: pass it as ``model`` to ``system_one``."""
    kind: str = "saved"
    """``"base"`` (one of the project's base models, listed first) or ``"saved"`` (saved by a run, or a managed
    project's version)."""
    project: str
    name: str
    """The base model id for a base model; the saved model's name otherwise. A managed project's versions are named
    ``YYYY-MM-DD-N``."""
    training_run_id: str | None = None
    base_model: str
    contract: str | None = None
    parent_model: str | None = None
    """The model the training run started from (``from_model``), if any."""
    step: int = 0
    has_optimizer: bool = False
    size_bytes: int | None = None
    expires_at: datetime | None = None
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    loss: float | None = None
    """The training loss when the model was saved: its run's ``train/loss`` at ``step``, or the last one before it.
    None for a base model, or when unknown."""
    accuracy: float | None = None
    """The training accuracy when the model was saved (``train/accuracy``, the same way as ``loss``)."""
    created_at: datetime


class DatasetInfo(_View):
    name: str
    project: str
    n_rows: int
    n_labelled_questions: int = 0
    question_types: dict[str, int] = Field(default_factory=dict)
    created_at: datetime


class Evaluation(_View):
    id: str
    project: str
    model: str
    dataset: str | None = None
    training_run_id: str | None = None
    source: str
    """``server`` or ``client``."""
    status: str
    results: dict[str, Any] | None = None
    created_at: datetime


class MetricPoint(_View):
    step: int
    key: str
    value: float
    source: str = "client"
    time: datetime | None = None


class DownloadedFile(_View):
    """A model file written by ``project.models.download``."""

    name: str
    path: Path
    size_bytes: int
    sha256: str  # of the bytes written, hex; checked against the API's when it gave one


class UsageRow(_View):
    """One hour of usage for one kind, project, base model, model and run."""

    hour: datetime
    kind: str
    """``decide`` (billed per 1M input tokens per base model) or ``train`` (billed per GPU-hour)."""
    project: str | None = None
    project_deleted: bool = False
    """The project was deleted (``project`` is None when its record is gone)."""
    base_model: str | None = None
    model: str | None = None
    """Decisions only: the saved model that answered (None for a base model: see ``base_model``)."""
    training_run_id: str | None = None
    """Training only."""
    input_tokens: int | None = None
    """Decisions only."""
    gpu_seconds: float | None = None
    """Training only: seconds of the run's reserved GPU, from request to release."""
    continuous_learning: bool | None = None
    """Decisions only: whether the project is managed, so billed at the continuous-learning price (managed and
    custom usage come as separate rows)."""
    price_per_mtok: float | None = None
    """Decisions only: the price per 1M input tokens applied."""
    base_cost_usd: float | None = None
    """Decisions only: the cost at the base price."""
    continuous_learning_cost_usd: float | None = None
    """Decisions only: the managed-project extra (0 in a custom project). ``cost_usd`` is the total."""
    cost_usd: float | None = None


class WhoAmI(_View):
    """The org and credential behind a request. ``user_id`` is None for an org API key."""

    org_id: str | None = None
    user_id: str | None = None
    key_id: str | None = None
    limits: dict[str, int | float] = Field(default_factory=dict)
    """The org's limits by name, for example ``max_projects_per_org`` or ``max_active_runs``. A count limit raises
    ``LimitReachedError`` once reached; a rate raises ``RateLimitError``: ``decisions_per_minute`` (60 by default,
    for the whole org) and the other ``..._per_minute``."""
    managed_projects: bool = False
    """Whether managed projects can be created here (they are coming soon where they can't)."""


class FutureInfo(_View):
    id: str
    kind: str | None = None
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
