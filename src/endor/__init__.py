"""Endor SDK: decisions with System One models, and fine-tuning on your own data.

import endor
from endor import Choice, Noul

client = endor.EndorClient()                                        # ENDOR_API_KEY
res = client.system_one({"body": "charged twice"},
                        {"dept": Choice(criteria={"billing": None, "technical": None}),
                         "urgent": Noul(instructions="Does this need an answer today?")},
                        model="tickets/v1")                         # a saved model, by its id
res.choices["dept"].choice, res.nouls["urgent"].noul

project = client.projects.get_or_create("tickets", base_models=["decider-2b"])  # "tickets/decider-2b" answers
with project.runs.create(rank=16) as run:   # on the project's first base model; closed on exit, even on errors
    steps = endor.Steps()                                           # the next step is sent before the last is read
    for batch in endor.data.batches(endor.data.rows_to_datums(rows), 16):
        steps.add(run.forward_backward(batch), run.optim_step(learning_rate=1e-4))
    steps.wait()
    model = run.save_checkpoint("v1").result()                      # "tickets/v1"
client.system_one(state, questions, model=model)

endor.user_report("what you did, what happened, what you expected", kind="bug")   # tell the Endor team
"""

from . import data, metrics, recipes, types
from ._retry import RetryPolicy
from ._user_report import user_report, user_report_async
from ._version import __version__
from .client import EndorClient
from .errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    DownloadError,
    EndorError,
    InsufficientBalanceError,
    InternalServerError,
    LimitReachedError,
    ModelRequiresProjectError,
    NoBaseModelError,
    NotFoundError,
    OperationFailedError,
    OverloadedError,
    PayloadTooLargeError,
    PermissionDeniedError,
    RateLimitError,
    ResponseValidationError,
    UnprocessableEntityError,
    WrongProjectKindError,
)
from .futures import APIFuture, Steps, gather, gather_async
from .images import Image
from .projects import Project
from .runs import Run
from .types import (
    AdamParams,
    BaseModelInfo,
    Choice,
    ChoiceAnswer,
    DatasetInfo,
    Datum,
    DecisionRow,
    DownloadedFile,
    Evaluation,
    ForwardOutput,
    ListModelsResponse,
    LoraConfig,
    LoraInfo,
    MetricPoint,
    ModelInfo,
    ModelMetadata,
    Noul,
    NoulAnswer,
    NoulCriteria,
    OptimStepOutput,
    ProjectInfo,
    RunInfo,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Target,
    Usage,
    UsageRow,
    UserReportKind,
    UserReportReceipt,
    WandbSettings,
    WhoAmI,
    answer_probabilities,
    option_keys,
    question_dict,
)

__all__ = [
    "__version__",
    "data",
    "metrics",
    "recipes",
    "types",
    "EndorClient",
    "Project",
    "Run",
    "APIFuture",
    "Steps",
    "gather",
    "gather_async",
    "RetryPolicy",
    "user_report",
    "user_report_async",
    "UserReportKind",
    "UserReportReceipt",
    # questions and answers
    "Image",
    "Noul",
    "Choice",
    "Score",
    "NoulCriteria",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "SystemOneResponse",
    "Usage",
    "ModelMetadata",
    "ListModelsResponse",
    "BaseModelInfo",
    # training
    "Datum",
    "DecisionRow",
    "Target",
    "AdamParams",
    "LoraConfig",
    "LoraInfo",
    "ForwardOutput",
    "OptimStepOutput",
    # resources
    "ProjectInfo",
    "RunInfo",
    "WandbSettings",
    "ModelInfo",
    "DatasetInfo",
    "Evaluation",
    "MetricPoint",
    "DownloadedFile",
    "UsageRow",
    "WhoAmI",
    "option_keys",
    "answer_probabilities",
    "question_dict",
    # errors
    "EndorError",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "InsufficientBalanceError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "LimitReachedError",
    "NoBaseModelError",
    "WrongProjectKindError",
    "DownloadError",
    "PayloadTooLargeError",
    "UnprocessableEntityError",
    "ModelRequiresProjectError",
    "RateLimitError",
    "OverloadedError",
    "InternalServerError",
    "ResponseValidationError",
    "APIConnectionError",
    "APITimeoutError",
    "OperationFailedError",
]
