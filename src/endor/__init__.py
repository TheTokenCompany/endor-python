"""Endor SDK: decisions with System One models, and fine-tuning on your own data.

import endor
from endor import Choice, Noul

client = endor.EndorClient()                                        # ENDOR_API_KEY
res = client.system_one({"body": "charged twice"},
                        {"dept": Choice(criteria={"billing": None, "technical": None}),
                         "urgent": Noul(instructions="Does this need an answer today?")},
                        model="tickets/v1")                         # a saved model, by its id
res.choices["dept"].choice, res.nouls["urgent"].noul

project = client.projects.get_or_create("tickets", base_model="decider-2b")   # "tickets" answers at once
with project.runs.create(base_model="decider-2b", rank=16) as run:   # closed on exit, even on errors
    for batch in endor.data.batches(endor.data.rows_to_datums(rows), 16):
        fb = run.forward_backward(batch)
        opt = run.optim_step(learning_rate=1e-4)
        endor.gather(fb, opt)
    model = run.save_checkpoint("v1").result()                      # "tickets/v1"
client.system_one(state, questions, model=model)
"""

from . import data, metrics, recipes, types
from ._retry import RetryPolicy
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
from .futures import APIFuture, gather, gather_async
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
    "gather",
    "gather_async",
    "RetryPolicy",
    # questions and answers
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
