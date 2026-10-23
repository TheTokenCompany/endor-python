"""Public data types.

Decisions
    Noul, Choice, Score          the three question types (plain dicts in the same shape are accepted too)
    NoulAnswer, ChoiceAnswer, ScoreAnswer, SystemOneResponse, Usage
    ModelMetadata, ListModelsResponse, BaseModelInfo, DecisionIndexInfo
    CorrectionInfo, QuestionCorrectionInfo, CorrectionAnswer   a decision's correction (``client.corrections``)

Training
    Datum        one state + one question + its Target (+ weight): the unit of ``forward`` / ``forward_backward``
    DecisionRow  one state + named questions + labels: the unit of datasets (README, "Data format")
    Target, LoraConfig, AdamParams, ForwardOutput, OptimStepOutput

Resources (read-only views of what the API returns)
    ProjectInfo, VersionInfo     a continuous learning project and its versions (``client.projects``)
    Learning, Training, Drift    a continuous learning project's learning (``project.learning()``)
    CustomModelInfo              a custom model (``client.custom_models``)
    RunInfo, DatasetInfo, Evaluation, MetricPoint, DownloadedFile, UsageRow, WhoAmI, FutureInfo

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
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

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
    "DecisionIndexInfo",
    "Target",
    "Datum",
    "DecisionRow",
    "LoraConfig",
    "LoraInfo",
    "AdamParams",
    "ForwardOutput",
    "OptimStepOutput",
    "ProjectInfo",
    "VersionInfo",
    "Learning",
    "Training",
    "Drift",
    "RunInfo",
    "CustomModelInfo",
    "DatasetInfo",
    "Evaluation",
    "MetricPoint",
    "DownloadedFile",
    "UsageRow",
    "WhoAmI",
    "UserReportKind",
    "UserReportReceipt",
    "CorrectionAnswer",
    "QuestionCorrection",
    "QuestionCorrectionInfo",
    "CorrectionInfo",
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

    id: str | None = None
    """The decision's id (``req_...``, the same as its ``X-Request-Id``): correct the decision with
    ``client.corrections.create(res, ...)``. None from a server that doesn't send one."""
    model: str
    """What answered: a project's version (``"tickets:v4"``, or ``"tickets"`` while its base model answers), a custom
    model (``"router-2"``) or a base model called directly (``"mapika/decider-2b"``)."""
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


class DecisionIndexInfo(BaseModel):
    """A base model's entry on the Jev Decision Index (``BaseModelInfo.decision_index``), the public board that runs
    every open Jev reproduction through the same public and private tests. The scores themselves are on ``url``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    rank: int | None = None
    """Its rank on the board, out of ``entrants`` (1 is best)."""
    entrants: int | None = None
    """How many entries the board ranks, Jev included."""
    tied: bool = False
    """Within the board's tie band of another entry, so it shares its rank (``=`` in ``endor base-models``)."""
    meant_for: list[str] = Field(default_factory=list)
    """The use cases it is among the best at, best first: ``"Code"``, ``"Agents"``, ``"Documents and email"`` and
    ``"Finance"`` when its mean rank on the index's benchmarks for one is in the top quarter of the board. Empty when
    none."""
    use_case_ranks: dict[str, float] = Field(default_factory=dict)
    """Its mean rank among the board's entrants on each use case's benchmarks (lower is better): how it compares with
    the other bases on the use case closest to yours."""
    task_ranks: dict[str, int] = Field(default_factory=dict)
    """Its rank among the board's entrants on each public benchmark, by what the benchmark tests, for example
    ``"phishing detection"`` (lower is better): the benchmarks most like your task say the most. A benchmark that
    needs more options than the base takes on Endor is left out."""
    note: str | None = None
    """Set where the index ran another form of the weights than Endor serves (for example a quantized one), so its
    scores can differ a little from Endor's answers."""
    name: str | None = None
    """Its name on the index."""
    engine: str | None = None
    """Its id on the index."""
    url: str | None = None
    """Its page on the index, with its scores on every benchmark."""
    source_url: str | None = None
    """Its weights or code, as the index links them."""
    edition: str | None = None
    """The index's edition, for example ``"Decision Index 0.3.1"``."""
    updated: str | None = None
    """The index's date (``YYYY-MM-DD``)."""


class BaseModelInfo(BaseModel):
    """A base decision model, called directly by its ``id`` (``client.system_one(..., model="mapika/decider-2b")``): a
    continuous learning project learns on one (``client.projects.create(name, base_model=id)``), and custom models are
    trained on one (``client.custom_models.create(name, base_model=id)``).

    To choose one, compare what it is meant for and its ranks (``decision_index``), its speed (``tokens_per_s``), its
    context (``max_question_tokens``), its options (``max_options``) and its price."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    """``owner/model``, as on Hugging Face: ``"mapika/decider-2b"``."""
    description: str | None = None
    release_date: str | None = None
    params: str | None = None
    """The model size, for example ``"26B (4B active)"``."""
    provider: str | None = None
    """Who published it."""
    leaderboard_rank: int | None = None
    """Its rank on the Jev Decision Index, when listed (``decision_index.rank``)."""
    decision_index: DecisionIndexInfo | None = None
    """Its entry on the Jev Decision Index: its rank, what it is meant for and its ranks per use case and per task.
    None when it isn't on the index."""
    tokens_per_s: int | None = None
    """Decision input tokens per second Endor served on it, where measured: a higher number answers sooner. None where
    it wasn't measured."""
    tokens_per_s_source: str | None = None
    """Where and how ``tokens_per_s`` was measured."""
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
    price_per_mtok_decide: float | None = None
    """USD per 1M decision input tokens of a custom model on this base (None while unpriced)."""
    price_per_mtok_decide_continuous_learning: float | None = None
    """The same for a continuous learning project on this base."""
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
    """Endor's metadata: ``kind`` and ``base_model``, and for a project's names the ``project``, the ``environment``
    and the ``model`` the name answers with now (``"tickets:v4"``, or ``"tickets"`` while its base model answers)."""

    @property
    def kind(self) -> str:
        """What the name calls, as Endor reports it: a continuous learning project (``"project"``) or a custom model in
        production (``"custom"``)."""
        if self.endor.get("kind"):
            return str(self.endor["kind"])
        return "project"

    @property
    def environment(self) -> str | None:
        """For a project's names: ``"production"`` (``"tickets"``, its production version) or ``"preview"``
        (``"tickets:preview"``, listed only while it has one). None for a custom model."""
        env = self.endor.get("environment")
        return str(env) if env else None


class ListModelsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    models: list[ModelMetadata]


# ---------------------------------------------------------------- training


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Target(_Strict):
    """What the loss compares against: a hard ``label`` (an option id) or soft ``probs`` over every option.

    ``Target(label="billing")`` or ``Target(probs={"billing": 0.7, "tech": 0.3})``; as a dict,
    ``{"probs": {...}}``. The bare map ``{"billing": 0.7, "tech": 0.3}`` is a row label's shape and raises
    ValueError; ``endor.data.label_target`` turns a row label into a ``Target``.
    """

    label: str | None = None
    probs: dict[str, float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _not_a_row_label(cls, v: Any) -> Any:
        bare_map = isinstance(v, Mapping) and v and "label" not in v and "probs" not in v
        if bare_map or isinstance(v, (str, int, float, list)):
            raise ValueError(
                'a Target is {"label": "billing"} or {"probs": {"billing": 0.7, "tech": 0.3}}; a bare probability '
                "map or an option key is a row label's shape (endor.data.label_target turns a row label into a Target)"
            )
        return v

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


class ProjectInfo(_View):
    """A continuous learning project as last fetched (``project.info_``)."""

    name: str
    description: str | None = None
    base_model: str
    """The base model the project learns on (``"mapika/decider-2b"``), chosen when it was created. It never changes."""
    production: str | None = None
    """The version in production, which ``"<project>"`` answers with (``"v4"``); None until a version goes to
    production, while the base model answers."""
    preview: str | None = None
    """The newest version while it isn't in production, which ``"<project>:preview"`` answers with. None when the
    newest version is in production (as it is while auto-promote is on, unless Endor held it back) and before the
    first version."""
    auto_promote: bool = True
    """On for a new project: each new version goes to production as soon as it is ready. Endor still holds back a first
    version that isn't clearly better than the base model, and a version whose agreement with the frontier model fell;
    those stay the preview. Off (``project.set_auto_promote(False)``): new versions land in the preview until you
    promote one. Promoting an older version turns it off (a rollback)."""
    promoted_at: datetime | None = None
    """When the production version last changed."""
    versions: int = 0
    """How many versions the project has."""
    created_at: datetime


class VersionInfo(_View):
    """One version of a continuous learning project (``project.versions()``, newest first)."""

    name: str
    """``"v1"``, ``"v2"``, ..."""
    base_model: str
    production: bool = False
    """It is in production: ``"<project>"`` answers with it."""
    preview: bool = False
    """It is the newest version and isn't in production, so ``"<project>:preview"`` answers with it. Never both."""
    accuracy: float | None = None
    """Its accuracy when it was trained, in percent (0 to 100, unlike custom models and evaluations, which give 0 to
    1). None: nothing to check on."""
    loss: float | None = None
    """Its mean cross-entropy on the same check (lower is better)."""
    size_bytes: int | None = None
    created_at: datetime


class Training(_View):
    """One training run of a continuous learning project."""

    at: datetime
    kept: bool
    """Saved as ``version`` (in production at once while auto-promote is on, unless ``held``; else the project's
    preview), or discarded (not better than the version it was compared with)."""
    version: str | None = None
    decisions: int | None = None
    """The project's decision count when the run started (None from a server that doesn't send it)."""
    labels: int | None = None
    """How many labels it had learned from when the run started (None as for ``decisions``)."""
    accuracy: float | None = None
    """Its accuracy in percent (0 to 100, as ``VersionInfo.accuracy``), with the 95% interval ``low`` to ``high``:
    agreement with a frontier model's answers, or with your held-back corrections once there are enough. None: nothing
    to check on."""
    low: float | None = None
    high: float | None = None
    checked: int | None = None
    """How many held-back answers ``accuracy`` was measured on (one per question a decision asked)."""
    loss: float | None = None
    """Its mean cross-entropy on the same decisions (lower is better)."""
    against: str | None = None
    """The version it was checked against (None: the base model)."""
    against_accuracy: float | None = None
    against_loss: float | None = None
    worse: list[str] = Field(default_factory=list)
    """The questions it was worse on."""
    held: str | None = None
    """Why a kept version stayed the preview instead of going to production with auto-promote on: a first version that
    wasn't clearly better than the base model, or one whose agreement with the frontier model fell against the
    production version's. None: it wasn't held."""


class Drift(_View):
    """Whether the served version's recent answers to one question differ from what it was checked on. Reported
    only: learning goes on as usual."""

    question: str
    changed: bool
    """The traffic moved enough to matter."""
    answer_shift: float | None = None
    """How far the mix of answers moved, 0 to 1."""
    confidence_change: float | None = None
    """The change in how sure it is (the likeliest answer's mean probability)."""


class Learning(_View):
    """A continuous learning project's learning (``project.learning()``). It is always on."""

    status: str
    """``learning``, ``training`` (a run is on), ``waiting`` (nothing to learn from yet), ``up_to_date`` (the model
    already answers as well as it can learn to), ``out_of_credit`` or ``limit_reached`` (the organization's daily
    learning limit; it goes on after midnight UTC). Treat unknown values as ``learning``."""
    progress: float = 0.0
    """The way to the next training run, 0 to 1."""
    serving: str
    """What ``"<project>"`` answers with now: ``"<project>:<version>"``, or ``"<project>"`` while its base model
    answers."""
    trained_at: datetime | None = None
    training_since: datetime | None = None
    """When the training run on now started (None unless ``status`` is ``training``)."""
    trainings: list[Training] = Field(default_factory=list)
    """The latest 50 training runs, newest first."""
    drift: list[Drift] = Field(default_factory=list)
    """Per question, whether recent traffic moved away from what the served version was checked on (empty until a
    version has served enough decisions)."""


class RunInfo(_View):
    id: str
    name: str | None = None
    """The run's own label."""
    model: str | None = None
    """The custom model name ``run.save_checkpoint()`` saves under when it is given no name (``client.runs.create(
    model=...)``, or ``client.custom_models.create(name, ...)``)."""
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
    """The custom model the run started from (``from_model``), if any."""
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


class CustomModelInfo(_View):
    """A custom model (``client.custom_models``): trained with the SDK and saved by a run. One in production is
    called by its name (``model="router-2"``); an archived one can be downloaded, evaluated, trained from and
    promoted, but decisions refuse it (``ModelNotInProductionError``)."""

    name: str
    """Its name, unique in the organization (with the continuous learning projects): ``model`` in ``system_one``
    while it is in production."""
    base_model: str
    production: bool = False
    """It is in production: decisions may call it."""
    state: str = "archived"
    """``"production"`` or ``"archived"`` (every save starts archived); ``custom_models.get`` says ``"saving"`` while
    the model's save runs. Treat unknown values as archived."""
    promoted_at: datetime | None = None
    """When it last went to production (None while archived)."""
    serving: str | None = None
    """In production, whether a decision server has it loaded: ``"deploying"`` right after a promotion (decisions are
    accepted; the first ones may wait while it loads), then ``"live"``, or ``"failed"`` when it couldn't be loaded
    (``serving_error`` says why; ``client.custom_models.promote(name)`` again retries). None while archived. Treat
    unknown values as ``"deploying"``."""
    serving_error: str | None = None
    """Why it couldn't be loaded, while ``serving`` is ``"failed"``; None otherwise."""
    contract: str | None = None
    training_run_id: str | None = None
    parent_model: str | None = None
    """The custom model its training run started from (``from_model``), if any."""
    step: int = 0
    has_optimizer: bool = False
    size_bytes: int | None = None
    expires_at: datetime | None = None
    """Archived models only: when it is deleted (a TTL). Models in production never expire."""
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    loss: float | None = None
    """The run's training loss at the model's step: its ``train/loss`` at ``step``, or the last one before it. None
    when unknown."""
    accuracy: float | None = None
    """The run's training accuracy at the model's step (``train/accuracy``, the same way as ``loss``), from 0 to 1."""
    created_at: datetime


class DatasetInfo(_View):
    """A dataset of decision rows for training and evaluating custom models (``client.datasets``)."""

    name: str
    n_rows: int
    n_labelled_questions: int = 0
    question_types: dict[str, int] = Field(default_factory=dict)
    created_at: datetime


class Evaluation(_View):
    id: str
    model: str
    """The custom model or base model it scored."""
    dataset: str | None = None
    """The dataset's name; None for rows sent inline and for client evaluations."""
    training_run_id: str | None = None
    source: str
    """``server`` or ``client``."""
    status: str
    results: dict[str, Any] | None = None
    """``n``, ``overall``, ``by_type`` and ``by_question`` (each with ``accuracy`` from 0 to 1, NLL, Brier, ECE, mean
    confidence and ``selective``) and ``rows_url``, once it is done (``client.evaluations.create``)."""
    rows_total: int | None = None
    """Server evaluations: its rows (the dataset's, or those sent inline)."""
    rows_done: int | None = None
    """Server evaluations: the rows decided so far, updated every few seconds while it runs."""
    started_at: datetime | None = None
    created_at: datetime


class MetricPoint(_View):
    step: int
    key: str
    value: float
    source: str = "client"
    time: datetime | None = None


class DownloadedFile(_View):
    """A model file written by ``client.custom_models.download`` or ``project.download``."""

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
    """The continuous learning project; None for a custom model's decisions, a base model called directly and
    training."""
    project_deleted: bool = False
    """The project was deleted (``project`` is None when its record is gone)."""
    base_model: str | None = None
    model: str | None = None
    """Decisions only: the project version (``"v4"``) or the custom model that answered (None for a base model: see
    ``base_model``)."""
    training_run_id: str | None = None
    """Training only."""
    input_tokens: int | None = None
    """Decisions only."""
    gpu_seconds: float | None = None
    """Training only: seconds of the run's reserved GPU, from request to release."""
    continuous_learning: bool | None = None
    """Decisions only: whether a continuous learning project answered, so billed at the continuous-learning price (a
    custom model's decisions come as separate rows)."""
    price_per_mtok: float | None = None
    """Decisions only: the price per 1M input tokens applied."""
    base_cost_usd: float | None = None
    """Decisions only: the cost at the base price."""
    continuous_learning_cost_usd: float | None = None
    """Decisions only: the continuous-learning extra (0 for a custom model). ``cost_usd`` is the total."""
    cost_usd: float | None = None


class WhoAmI(_View):
    """The org and credential behind a request. ``user_id`` is None for an org API key."""

    org_id: str | None = None
    user_id: str | None = None
    key_id: str | None = None
    limits: dict[str, int | float] = Field(default_factory=dict)
    """The org's limits by name, for example ``max_projects_per_org``, ``max_production_models_per_org`` (custom
    models in production), ``max_custom_models_per_org``, ``max_datasets_per_org`` or ``max_active_runs`` (runs that
    aren't closed, idle ones included; 25 by default). A count limit raises ``LimitReachedError`` once reached
    (``ProductionFullError`` for the production slots); a rate raises ``RateLimitError``: ``decisions_per_minute``
    (1,200 by default, for the whole org) and the other ``..._per_minute``. ``endor whoami`` lists them."""
    continuous_learning: bool = False
    """Whether the org has continuous learning projects (a per-organization switch, off until we turn it on). While
    off, ``client.projects`` and decisions on a project raise ``PermissionDeniedError`` (403
    ``continuous_learning_not_enabled``); custom models work for every organization."""


UserReportKind = Literal["bug", "confusing", "works"]
"""What a report says: ``bug`` (Endor did something unexpected), ``confusing`` (the docs, a message or the API are
unclear) or ``works`` (an integration runs as it should)."""


class UserReportReceipt(_View):
    """The API's answer to a report (``endor.user_report``, ``client.user_report``)."""

    id: str
    """The report's id (``urep_...``)."""
    message: str
    """A thank-you from the Endor team."""


CorrectionAnswer = bool | int | str
"""One question's answer, exactly one of its options as the decision asked it: a choice's option id, a noul's ``True``
or ``False``, a score's level index (0 is the first level)."""


class _QuestionCorrectionBase(TypedDict):
    correct: bool


class QuestionCorrection(_QuestionCorrectionBase, total=False):
    """One question of a correction you send: ``{"correct": True}``, ``{"correct": False, "answer": ...}`` or
    ``{"correct": False}``."""

    answer: CorrectionAnswer


class QuestionCorrectionInfo(_View):
    """One question of a decision's correction."""

    correct: bool
    """Whether the model answered it right."""
    answer: CorrectionAnswer | None = None
    """The right answer: what was served when it was right; else the answer sent, or the only other option of a noul
    or of a choice or score of two; else None (only known to be wrong, never learned from)."""
    served: CorrectionAnswer
    """What Endor answered."""


class CorrectionInfo(_View):
    """A decision's correction (``client.corrections``)."""

    request_id: str
    """The decision's id (``req_...``)."""
    project: str | None = None
    """The continuous learning project that made the decision; None for a custom model's decision."""
    questions: dict[str, QuestionCorrectionInfo]
    """The corrected questions, by name."""
    comment: str | None = None
    created_at: datetime


class FutureInfo(_View):
    id: str
    kind: str | None = None
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
