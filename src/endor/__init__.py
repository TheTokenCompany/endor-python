"""Endor SDK: decision models that learn from your decisions, and custom models you train yourself.

import endor
from endor import Choice, Noul

client = endor.EndorClient()                                        # ENDOR_API_KEY
project = client.projects.get_or_create("tickets", base_model="decider-2b")    # continuous learning
res = client.system_one({"body": "charged twice"},
                        {"dept": Choice(criteria={"billing": None, "technical": None}),
                         "urgent": Noul(instructions="Does this need an answer today?")},
                        model="tickets")                            # the project's production version
res.choices["dept"].choice, res.nouls["urgent"].noul
client.corrections.create(res, {"urgent": {"correct": False}})     # it was wrong: the project learns from it

# Advanced: a custom model you train yourself, called by its name once promoted
with client.custom_models.create("router-2", base_model="decider-2b", rank=16) as run:
    steps = endor.Steps()                                           # the next step is sent before the last is read
    for batch in endor.data.batches(endor.data.rows_to_datums(rows), 16):
        steps.add(run.forward_backward(batch), run.optim_step(learning_rate=1e-4))
    steps.wait()
    run.save_checkpoint().result()                                  # "router-2", in the archive
client.custom_models.promote("router-2")
client.system_one(state, questions, model="router-2")

endor.user_report("what you did, what happened, what you expected", kind="bug")   # tell the Endor team
"""

from . import data, metrics, recipes, types
from ._retry import RetryPolicy
from ._user_report import user_report, user_report_async
from ._version import __version__
from .client import EndorClient
from .custom_models import CustomModels
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
    ModelInProductionError,
    ModelNotInProductionError,
    ModelRequiresProjectError,
    NoPreviewError,
    NotFoundError,
    OperationFailedError,
    OverloadedError,
    PayloadTooLargeError,
    PermissionDeniedError,
    ProductionFullError,
    RateLimitError,
    ResponseValidationError,
    UnprocessableEntityError,
    VersionNotCallableError,
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
    CorrectionAnswer,
    CorrectionInfo,
    CustomModelInfo,
    DatasetInfo,
    Datum,
    DecisionIndexInfo,
    DecisionRow,
    DownloadedFile,
    Drift,
    Evaluation,
    ForwardOutput,
    Learning,
    ListModelsResponse,
    LoraConfig,
    LoraInfo,
    MetricPoint,
    ModelMetadata,
    Noul,
    NoulAnswer,
    NoulCriteria,
    OptimStepOutput,
    ProjectInfo,
    QuestionCorrection,
    QuestionCorrectionInfo,
    RunInfo,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Target,
    Training,
    Usage,
    UsageRow,
    UserReportKind,
    UserReportReceipt,
    VersionInfo,
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
    "CustomModels",
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
    "CorrectionAnswer",
    "QuestionCorrection",
    "QuestionCorrectionInfo",
    "CorrectionInfo",
    "ModelMetadata",
    "ListModelsResponse",
    "BaseModelInfo",
    "DecisionIndexInfo",
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
    "VersionInfo",
    "Learning",
    "Training",
    "Drift",
    "CustomModelInfo",
    "RunInfo",
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
    "ProductionFullError",
    "ModelNotInProductionError",
    "ModelInProductionError",
    "VersionNotCallableError",
    "NoPreviewError",
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
