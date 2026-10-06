"""Public data types.

Decisions
    Noul, Choice, Score          the three question types (plain dicts in the same shape are accepted too)
    NoulAnswer, ChoiceAnswer, ScoreAnswer, SystemOneResponse, Usage
    ModelMetadata, ListModelsResponse, BaseModelInfo

Training
    Datum        one state + one question + its Target (+ weight): the unit of ``forward`` / ``forward_backward``
    DecisionRow  one state + named questions + labels: the unit of datasets (docs/DATA_FORMAT.md)
    Target, LoraConfig, AdamParams, ForwardOutput, OptimStepOutput

Resources (read-only views of what the API returns)
    ProjectInfo, RunInfo, ModelInfo, DatasetInfo, Evaluation, MetricPoint, ArchiveInfo, UsageRow, WhoAmI, FutureInfo

Option ids are the only identifiers used for a question's options anywhere in the SDK:
    noul   -> "false", "true"
    choice -> the criteria keys, in the order given
    score  -> "0" .. "n-1" (the level index)
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from functools import cached_property
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
    "AdamParams",
    "ForwardOutput",
    "OptimStepOutput",
    "ProjectInfo",
    "RunInfo",
    "ModelInfo",
    "DatasetInfo",
    "Evaluation",
    "MetricPoint",
    "ArchiveInfo",
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
"""A row label in the natural form for its question type (docs/DATA_FORMAT.md)."""

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

    ```python
    Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"])
    ```
    """

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: list[JSONContent]
    """Level descriptions, lowest first. 2 to 10 levels; level ``i`` has option id ``str(i)``."""

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
    """The model that answered: a base id, or ``"<project>/<name>"``."""
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
    """A base decision model you can decide with and fine-tune."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    description: str | None = None
    release_date: str | None = None
    max_options: int
    """The most options a Choice (or levels a Score) may have on this base."""
    trainable: bool = True
    default_rank: int = 16
    lora_targets: list[str] = Field(default_factory=list)
    price_per_mtok_train: float | None = None
    price_per_mtok_decide: float | None = None


class ModelMetadata(BaseModel):
    """One entry of ``client.models.list()``: a base model or one of your saved models."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str
    """The id to pass as ``model``."""
    description: str = ""
    release_date: str | None = None
    endor: dict[str, Any] = Field(default_factory=dict)
    """Endor's metadata: ``kind`` (``base`` or ``model``), and for saved models ``project``, ``base_model``,
    ``training_run_id`` and ``step``."""

    @property
    def kind(self) -> str:
        return str(self.endor.get("kind") or ("model" if "/" in self.name else "base"))


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
    """One dataset row: a decision request plus labels (docs/DATA_FORMAT.md)."""

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
    model_config = ConfigDict(extra="ignore")


class ForwardOutput(_View):
    """The result of ``forward`` or ``forward_backward``."""

    outputs: list[dict[str, Any]]
    """Per datum, in input order: ``{"probabilities": {option_id: p}, "loss": float}``."""
    metrics: dict[str, float]
    """``loss:sum``, ``weight:sum``, ``loss:mean``, ``accuracy``, ``n``."""

    @property
    def probabilities(self) -> list[dict[str, float]]:
        return [dict(o["probabilities"]) for o in self.outputs]

    @property
    def losses(self) -> list[float]:
        return [float(o["loss"]) for o in self.outputs]

    @property
    def loss(self) -> float:
        """The weighted mean loss over the batch."""
        return self.metrics.get("loss:mean", float("nan"))

    @property
    def accuracy(self) -> float:
        return self.metrics.get("accuracy", float("nan"))


class OptimStepOutput(_View):
    step: int
    grad_norm: float
    learning_rate: float


class ProjectInfo(_View):
    name: str
    description: str | None = None
    n_datasets: int = 0
    n_runs: int = 0
    n_models: int = 0
    created_at: datetime


class RunInfo(_View):
    id: str
    project: str
    name: str | None = None
    base_model: str
    lora: LoraConfig = Field(default_factory=LoraConfig)
    status: str
    """``provisioning``, ``ready``, ``idle``, ``closed`` or ``failed``."""
    ready_future_id: str | None = None
    step: int = 0
    next_seq_id: int = 0
    parent_model: str | None = None
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    code_hash: str | None = None
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    failure: dict[str, Any] | None = None
    created_at: datetime


class ModelInfo(_View):
    id: str
    """``"<project>/<name>"``: pass it as ``model`` to ``system_one``."""
    project: str
    name: str
    training_run_id: str | None = None
    base_model: str
    step: int = 0
    has_optimizer: bool = False
    size_bytes: int | None = None
    expires_at: datetime | None = None
    user_metadata: dict[str, Any] = Field(default_factory=dict)
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


class ArchiveInfo(_View):
    """A short-lived download link for a saved model's files."""

    url: str
    expires_at: datetime
    files: list[str] = Field(default_factory=list)


class UsageRow(_View):
    hour: datetime
    kind: str
    """``train``, ``decide`` or ``storage``."""
    project: str | None = None
    base_model: str | None = None
    training_run_id: str | None = None
    tokens: int = 0
    cost_usd: float = 0.0


class WhoAmI(_View):
    user_id: str
    key_id: str
    email: str | None = None
    key_prefix: str | None = None


class FutureInfo(_View):
    id: str
    kind: str | None = None
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
