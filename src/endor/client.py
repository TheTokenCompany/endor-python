"""EndorClient: decisions, plus fine-tuning.

    client = endor.EndorClient()                               # ENDOR_API_KEY from the environment
    client.system_one(state, questions, model="tickets/v1")    # a decision with one saved model
    client.models.list()                                       # every name you can pass as model
    client.base_models()                                       # the base models, with limits and prices
    project = client.projects.get_or_create("tickets", base_model="decider-2b")   # datasets, runs, models

``model`` always names a project: ``"<project>/<name>"`` (one saved model), ``"<project>/base"`` (its base model) or
``"<project>"`` (a managed project's newest version, else its base model). A bare base model id is refused
(``ModelRequiresProjectError``).

Environment: ``ENDOR_API_KEY``, ``ENDOR_BASE_URL``, ``ENDOR_DEFAULT_MODEL``, ``ENDOR_LOG_LEVEL``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, TypeVar, overload

import httpx
from pydantic import BaseModel, ValidationError

from . import _constants as C
from ._http import Transport
from ._retry import RetryPolicy
from .errors import EndorError, NotFoundError, ResponseValidationError
from .projects import Projects
from .types import (
    BaseModelInfo,
    JSONContent,
    ListModelsResponse,
    Question,
    SystemOneResponse,
    UsageRow,
    WhoAmI,
    question_dict,
)

__all__ = ["EndorClient", "Models", "DEFAULT_BASE_URL"]

DEFAULT_BASE_URL = C.DEFAULT_BASE_URL

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class EndorClient:
    """The entry point. One instance per process is plenty; it is safe to share across threads."""

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
    ) -> None:
        """Create a client.

        Args:
            api_key: Your API key (``edk_...``). Defaults to ``ENDOR_API_KEY``.
            model: Default model for ``system_one``, for example ``"tickets"``. Defaults to ``ENDOR_DEFAULT_MODEL``;
                without either, each ``system_one`` call must pass ``model``.
            retry: A ``RetryPolicy``; ``RetryPolicy(max_retries=0)`` disables retries.
            timeout: Seconds per HTTP request. A future poll waits up to 25 s on the server on top of this.
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
        )
        self.models = Models(self._t)
        """``client.models.list()``: every name you can pass as ``model``."""
        self.projects = Projects(self._t, capture)
        """``client.projects``: create, fetch and list projects."""

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
            state: Text, a JSON object or an array: the data the decisions are about.
            questions: 1 to 64 named questions (``Noul``, ``Choice``, ``Score`` or dicts in the same shape).
            model: ``"<project>/<name>"`` (a saved model), ``"<project>/base"`` (its base model) or ``"<project>"``
                (a managed project's newest version, else its base model). Defaults to the client's model.
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
            timeout=timeout,
            retry=retry,
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
            timeout=timeout,
            retry=retry,
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
                'no model: pass model="<project>/<name>", "<project>/base" or "<project>", '
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

    def base_models(self) -> list[BaseModelInfo]:
        """The base models you can fine-tune and decide with (through a project, as ``"<project>/base"``): option
        limits, ``hf_repo``, ``contract`` and prices (``GET /v1/base_models``)."""
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
        """Every name you can pass as ``model``: ``"<project>"`` (a managed project's newest version, else the base)
        and ``"<project>/base"`` for each project with a base model, then ``"<project>/<name>"`` for each saved,
        unexpired model. ``.kind`` is ``project``, ``base`` or ``model``. The base model catalog is
        ``client.base_models()``."""
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
