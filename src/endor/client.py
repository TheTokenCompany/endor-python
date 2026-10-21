"""EndorClient: decisions, plus fine-tuning.

    client = endor.EndorClient()                               # ENDOR_API_KEY from the environment
    client.system_one(state, questions, model="tickets/v1")    # a decision with one saved model
    client.predict("tickets/v1", rows)                         # the answers for many rows, in their order
    client.models.list()                                       # every name you can pass as model
    client.base_models()                                       # the base models, with limits and prices
    project = client.projects.get_or_create("tickets", base_models=["decider-2b"])   # datasets, runs, models
    client.corrections.create(res, {"urgent": {"correct": False}})   # a decision was wrong: its project learns
    client.user_report("what you did, what happened, what you expected", kind="bug")      # tell the Endor team

``model`` always names a project: ``"<project>/<name>"`` (one saved model), ``"<project>/<base id>"`` (one of its base
models) or ``"<project>"`` (a continuous learning project's newest version, else its first base model). A bare base
model id is refused (``ModelRequiresProjectError``).

Environment: ``ENDOR_API_KEY``, ``ENDOR_BASE_URL``, ``ENDOR_DEFAULT_MODEL``, ``ENDOR_LOG_LEVEL``,
``ENDOR_REPORT_LATENCY``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, TypeVar, overload

import httpx
from pydantic import BaseModel, ValidationError

from . import _constants as C
from . import _predict, _user_report
from ._corrections import Corrections
from ._http import Transport
from ._images import Images
from ._retry import RetryPolicy
from .errors import EndorError, NotFoundError, ResponseValidationError
from .projects import Projects
from .types import (
    Answer,
    BaseModelInfo,
    JSONContent,
    ListModelsResponse,
    Question,
    SystemOneResponse,
    UsageRow,
    UserReportKind,
    UserReportReceipt,
    WhoAmI,
    question_dict,
)

__all__ = ["EndorClient", "Models", "DEFAULT_BASE_URL"]

DEFAULT_BASE_URL = C.DEFAULT_BASE_URL

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class EndorClient:
    """The entry point. One instance per process is plenty; it is safe to share across threads.

    When Endor doesn't do what you (or your coding agent) expected, or something is unclear, tell the Endor team with
    ``client.user_report("what you did, what happened, what you expected", kind="bug")``; ``kind="works"`` once an
    integration runs. Every error's ``str()`` ends with that hint, and ``endor.user_report`` works without a client
    or key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
        base_url: str | None = None,
        capture: bool = True,
        exclude_from_training: bool = False,
        gzip: bool = True,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        report_latency: bool | None = None,
    ) -> None:
        """Create a client.

        Args:
            api_key: Your API key (``edk_...``). Defaults to ``ENDOR_API_KEY``.
            model: Default model for ``system_one``, for example ``"tickets"``. Defaults to ``ENDOR_DEFAULT_MODEL``;
                without either, each ``system_one`` call must pass ``model``.
            retry: A ``RetryPolicy``; ``RetryPolicy(max_retries=0)`` disables retries. Without one, decisions get a
                total budget of 660 s instead of 60 s, so a cold start and its retry fit.
            timeout: Seconds per HTTP request (default 30). A future poll waits up to 25 s on the server on top of this.
                Without one, decisions wait up to 330 s, so a model that has to start (a few minutes) still answers.
            headers: Extra headers for every request.
            base_url: API root. Defaults to ``ENDOR_BASE_URL``, then production.
            capture: Send each run's settings and the current git commit (hash and dirty flag only, never
                file contents) so the dashboard can show how a model was made.
            exclude_from_training: Keep this client's decisions out of continuous learning by default (benchmarks,
                evaluations). They are still answered, billed and logged. ``system_one`` can override it per call.
            gzip: Compress request bodies over 32 KB (large dataset uploads and batches). On by default.
            http_client, async_http_client: Your own ``httpx`` clients, if you need one connection pool or
                custom settings. They are not closed with this client.
            transport, async_transport: Custom ``httpx`` transports (mostly for tests).
            report_latency: Tell Endor how long each request took for you, from sending it to reading the response,
                with the next request (``X-Endor-Client-Latency``), so the dashboard shows your latency next to
                Endor's. Nothing waits for it. Defaults to on, unless ``ENDOR_REPORT_LATENCY`` is ``0``.

        Raises:
            EndorError: No API key was given and ``ENDOR_API_KEY`` is not set.
        """
        if transport is not None and http_client is not None:
            raise ValueError("transport and http_client are mutually exclusive")
        key = (api_key if api_key is not None else _env(C.API_KEY_ENV) or "").strip()
        if not key:
            raise EndorError(f"no API key: pass api_key=... or set {C.API_KEY_ENV}")
        if not key.isascii() or not key.isprintable() or " " in key:
            raise EndorError("the API key must be printable ASCII without whitespace")
        self.api_key = key
        self.base_url = (base_url or _env(C.BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
        self.default_model: str | None = model or _env(C.DEFAULT_MODEL_ENV)
        self.timeout = C.DEFAULT_TIMEOUT if timeout is None else float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.retry = retry or RetryPolicy()
        # decisions wait out a cold start, unless the caller chose a timeout or a retry policy
        self._decision_timeout = C.DECISION_TIMEOUT if timeout is None else self.timeout
        self._decision_retry = retry or RetryPolicy(timeout=C.DECISION_RETRY_BUDGET)
        self.exclude_from_training = bool(exclude_from_training)
        """The default for ``system_one(exclude_from_training=...)``."""
        self._t = Transport(
            self.api_key,
            self.base_url,
            retry=self.retry,
            timeout=self.timeout,
            headers=headers,
            gzip_bodies=gzip,
            http_client=http_client,
            async_http_client=async_http_client,
            transport=transport,
            async_transport=async_transport,
            report_latency=report_latency,
        )
        self.models = Models(self._t)
        """``client.models.list()``: every name you can pass as ``model``."""
        self.images = Images(self._t)
        """``client.images.upload(...)``: images for training data, uploaded once and referenced by hash (the
        datasets, runs and predictions below do it for the ``endor.Image`` objects in your rows and datums)."""
        self.projects = Projects(self._t, capture, self.images)
        """``client.projects``: create, fetch and list projects."""
        self.corrections = Corrections(self._t)
        """``client.corrections``: whether past decisions were right and, if not, the right answers (``create``,
        ``batch``, ``get``, ``delete``). Continuous learning projects learn from them."""

    def __repr__(self) -> str:
        return f"EndorClient(base_url={self.base_url!r}, model={self.default_model!r})"

    # ------------------------------------------------------------------ decisions
    @overload
    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        exclude_from_training: bool | None = None,
        response_model: None = None,
    ) -> SystemOneResponse: ...

    @overload
    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        exclude_from_training: bool | None = None,
        response_model: type[ResponseT],
    ) -> ResponseT: ...

    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        exclude_from_training: bool | None = None,
        response_model: type[ResponseT] | None = None,
    ) -> SystemOneResponse | ResponseT:
        """Answer named questions about a state with one forward pass per question.

        Args:
            state: Text, a JSON object or an array: the data the decisions are about. On a base that reads images,
                it may hold ``endor.Image`` objects (as the state, a list item or an object value).
            questions: 1 to 64 named questions (``Noul``, ``Choice``, ``Score`` or dicts in the same shape).
            model: ``"<project>/<name>"`` (a saved model), ``"<project>/<base id>"`` (one of its base models) or
                ``"<project>"`` (a continuous learning project's newest version, else its first base model).
                Defaults to the client's model.
            retry, timeout: Overrides for this call.
            extra_headers: Extra request headers.
            extra_body: Extra top-level request fields, merged last.
            exclude_from_training: Keep this decision out of continuous learning (a benchmark or an evaluation); it
                is still answered, billed and logged. None: the client's ``exclude_from_training``.
            response_model: A ``SystemOneResponse`` subclass with one typed attribute per question.

        Returns:
            A ``SystemOneResponse`` (``.answers``, ``.nouls``, ``.choices``, ``.scores``, ``.model``, ``.usage``), or
            an instance of ``response_model``.

        Raises:
            EndorError: No ``model`` was given and the client has no default model.
            APIError: ``NotFoundError`` for an unknown model, ``ModelRequiresProjectError`` for a bare base model
                id, ``NoBaseModelError`` for a project without a base model, ``UnprocessableEntityError`` naming a
                bad question, ``RateLimitError`` and ``OverloadedError`` when told to wait.
        """
        body = self._decision_body(state, questions, model, extra_body, exclude_from_training)
        raw = self._t.request(
            "POST",
            "/v1/systemone",
            json=body,
            method_name="client.system_one",
            timeout=self._decision_timeout if timeout is None else timeout,
            retry=retry or self._decision_retry,
            extra_headers=extra_headers,
        )
        return _parse_decision(response_model, raw)

    async def system_one_async(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        exclude_from_training: bool | None = None,
        response_model: type[ResponseT] | None = None,
    ) -> SystemOneResponse | ResponseT:
        """``system_one`` for async code."""
        body = self._decision_body(state, questions, model, extra_body, exclude_from_training)
        raw = await self._t.arequest(
            "POST",
            "/v1/systemone",
            json=body,
            method_name="client.system_one",
            timeout=self._decision_timeout if timeout is None else timeout,
            retry=retry or self._decision_retry,
            extra_headers=extra_headers,
        )
        return _parse_decision(response_model, raw)

    def _decision_body(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        model: str | None,
        extra_body: Mapping[str, Any] | None,
        exclude_from_training: bool | None = None,
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("at least one question is required")
        if len(questions) > C.MAX_QUESTIONS_PER_REQUEST:
            raise ValueError(f"at most {C.MAX_QUESTIONS_PER_REQUEST} questions per request")
        model = model or self.default_model
        if not model and not (extra_body and extra_body.get("model")):
            raise EndorError(
                'no model: pass model="<project>/<name>", "<project>/<base id>" or "<project>", '
                f"or set a default with EndorClient(model=...) or {C.DEFAULT_MODEL_ENV}"
            )
        body: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": {name: question_dict(q) for name, q in questions.items()},
        }
        exclude = self.exclude_from_training if exclude_from_training is None else exclude_from_training
        if exclude:  # only when on, so every other body stays exactly TypeSafe's
            body["exclude_from_training"] = True
        if extra_body:
            body.update(extra_body)
        return body

    # ------------------------------------------------------------------ predictions
    def predict(
        self, model: str, rows: Iterable[Any], *, progress: bool = True, timeout: float | None = None
    ) -> list[dict[str, Answer]]:
        """The answers to every question of every row, in the rows' order: one server-side evaluation of the rows
        without their labels, so nothing needs uploading and nothing is scored. For many rows at once (new traffic,
        a backfill) instead of a ``system_one`` call per row; billed as decisions, the same per row.

        Args:
            model: ``"<project>/<name>"``, ``"<project>/<base id>"`` or ``"<project>"``, as for ``system_one``; the
                evaluation is made in that project (a custom project).
            rows: Decision rows in any shape ``project.datasets.upload`` takes (``{"state", "questions"}``; labels
                are ignored), at most 50,000 and 32 MB per call.
            progress: Print a progress line to stderr every 10 s while the evaluation runs
                (``evaluation evl_...: 1,200/10,000 rows (12%) · ~2 min left``).
            timeout: Seconds to wait at most (``TimeoutError`` after them; the evaluation keeps running). None
                waits until it is done.

        Returns:
            One dict per row, in order: question name → ``NoulAnswer``, ``ChoiceAnswer`` or ``ScoreAnswer``, as
            ``system_one(...).answers`` has them.

        Raises:
            ValueError: No rows, or more than 50,000.
            APIError: As for ``project.evaluate``: ``NotFoundError`` for an unknown project or model,
                ``UnprocessableEntityError`` naming a bad row (``rows.<i>...``) or a question the model can't read,
                ``WrongProjectKindError`` for a continuous learning project, ``RateLimitError``.
            OperationFailedError: The evaluation failed (for example a row too long for the model,
                ``context_too_long``).
        """
        project, body, n = _predict.prediction(model, rows)
        self.images.prepare_states(body["rows"])  # images uploaded once, sent as references
        r = self._t.request(
            "POST",
            f"/v1/projects/{project}/evaluations",
            json=body,
            method_name="client.predict",
            idempotent=True,
        )
        ev = _predict.wait(self._t, r, progress, timeout, "client.predict")
        return _predict.answers(self._t, ev, n)

    async def predict_async(
        self, model: str, rows: Iterable[Any], *, progress: bool = True, timeout: float | None = None
    ) -> list[dict[str, Answer]]:
        """``predict`` for async code."""
        project, body, n = _predict.prediction(model, rows)
        self.images.prepare_states(body["rows"])
        r = await self._t.arequest(
            "POST",
            f"/v1/projects/{project}/evaluations",
            json=body,
            method_name="client.predict",
            idempotent=True,
        )
        ev = await _predict.wait_async(self._t, r, progress, timeout, "client.predict")
        return await _predict.answers_async(self._t, ev, n)

    def base_models(self) -> list[BaseModelInfo]:
        """The base models you can fine-tune and decide with (through a project that has it, as
        ``"<project>/<id>"``): option limits, ``hf_repo``, ``contract`` and prices (``GET /v1/base_models``)."""
        try:
            r = self._t.request("GET", "/v1/base_models", method_name="client.base_models")
            return [_parse(BaseModelInfo, b, "GET /v1/base_models") for b in r.get("base_models", [])]
        except NotFoundError:  # an API from before /v1/base_models: the catalog was part of /v1/models
            r = self._t.request("GET", "/v1/models", method_name="client.base_models")
        out = []
        for m in r.get("models", []):
            meta = m.get("endor") or {}
            if meta.get("kind", "base" if "/" not in m.get("name", "") else "model") != "base":
                continue
            out.append(
                BaseModelInfo.model_validate(
                    {
                        "id": m["name"],
                        "description": m.get("description"),
                        "release_date": m.get("release_date"),
                        **meta,
                    }
                )
            )
        return out

    # ------------------------------------------------------------------ account
    def whoami(self) -> WhoAmI:
        """The org and key behind this client's credentials (``user_id`` is None for an org API key)."""
        return _parse(WhoAmI, self._t.request("GET", "/v1/whoami", method_name="client.whoami"), "GET /v1/whoami")

    def balance(self) -> float | None:
        """The org's prepaid balance in USD: credits bought or granted, minus usage so far. At $0 or less, paid calls
        raise ``InsufficientBalanceError``. None on a server without billing (a local API)."""
        r = self._t.request("GET", "/v1/balance", method_name="client.balance")
        return None if r["balance_usd"] is None else float(r["balance_usd"])

    def usage(self, starting_on: datetime, ending_before: datetime, project: str | None = None) -> list[UsageRow]:
        """Hourly usage with cost, for up to 14 days: ``decide`` rows (input tokens) and ``train`` rows
        (GPU-seconds). Datetimes without a timezone are taken as UTC."""
        params = {
            "starting_on": _utc(starting_on).isoformat(),
            "ending_before": _utc(ending_before).isoformat(),
            "project": project,
        }
        r = self._t.request("GET", "/v1/usage", params=params, method_name="client.usage")
        return [_parse(UsageRow, u, "GET /v1/usage") for u in r]

    # ------------------------------------------------------------------ reports
    def user_report(
        self,
        message: str,
        *,
        kind: UserReportKind = "bug",
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> UserReportReceipt:
        """Tell the Endor team that something didn't work as expected (``kind="bug"``), was unclear
        (``"confusing"``), or works (``"works"``): ``POST /v1/user_reports``, with this client's key so the team sees
        your org.

        Args:
            message: What you did, what happened and what you expected (up to 4000 characters; longer is cut).
            kind: ``"bug"``, ``"confusing"`` or ``"works"``.
            request_id: The ``req_...`` id of the request it is about (an error's ``request_id``). Default: this
                client's latest failed request (for ``"works"``, its latest request).
            context: Extra details as short values: at most 30 keys in all, strings cut at 500 characters. The SDK
                adds its version, Python's, the platform and the latest failure's status, code and endpoint; never
                your key, a request body or a decision's content.

        Returns:
            A ``UserReportReceipt`` with the report's ``id``.

        Raises:
            ValueError: An empty message, or an unknown ``kind``.
            RateLimitError: Too many reports for now (``retry_after``); never retried.
        """
        return _user_report.send(self._t, message, kind, request_id, context, "client.user_report")

    async def user_report_async(
        self,
        message: str,
        *,
        kind: UserReportKind = "bug",
        request_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> UserReportReceipt:
        """``report`` for async code."""
        return await _user_report.send_async(self._t, message, kind, request_id, context, "client.user_report")

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Close the HTTP connections this client opened."""
        self._t.close()

    async def aclose(self) -> None:
        """Close the HTTP connections this client opened (async)."""
        await self._t.aclose()

    def __enter__(self) -> EndorClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> EndorClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class Models:
    """``client.models``: the model catalog."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def list(
        self,
        *,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ListModelsResponse:
        """Every name you can pass as ``model``: for each project with a base model, ``"<project>"`` (a continuous
        learning project's newest version, else its first base model) and ``"<project>/<base id>"`` for each of its
        base models; then ``"<project>/<name>"`` for each saved, unexpired model. ``.kind`` is ``project``, ``base`` or
        ``model``. The base model catalog is ``client.base_models()``."""
        raw = self._t.request(
            "GET",
            "/v1/models",
            method_name="client.models.list",
            retry=retry,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        return _parse(ListModelsResponse, raw, "GET /v1/models")

    async def list_async(
        self,
        *,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ListModelsResponse:
        """``list()`` for async code."""
        raw = await self._t.arequest(
            "GET",
            "/v1/models",
            method_name="client.models.list",
            retry=retry,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        return _parse(ListModelsResponse, raw, "GET /v1/models")


def _parse_decision(response_model: type[ResponseT] | None, raw: Any) -> SystemOneResponse | ResponseT:
    """Parse a decision response. A ``SystemOneResponse`` subclass gets each declared question field filled from
    ``answers`` (``class Routed(SystemOneResponse): dept: ChoiceAnswer``)."""
    endpoint = "POST /v1/systemone"
    if response_model is None:
        return _parse(SystemOneResponse, raw, endpoint)
    if isinstance(raw, dict) and isinstance(raw.get("answers"), dict) and issubclass(response_model, SystemOneResponse):
        own = set(response_model.model_fields) - set(SystemOneResponse.model_fields)
        lifted = {name: raw["answers"][name] for name in own if name in raw["answers"] and name not in raw}
        if lifted:
            raw = {**raw, **lifted}
    return _parse(response_model, raw, endpoint)


def _parse(model: type[ResponseT], raw: Any, endpoint: str) -> ResponseT:
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        first = e.errors(include_url=False)[0]
        path = ".".join(str(p) for p in first.get("loc", ())) or "<body>"
        raise ResponseValidationError(200, raw, None, f"{path}: {first.get('msg')}", endpoint) from e


def _utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


def _env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None
