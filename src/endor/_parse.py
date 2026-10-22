"""Response bodies as the SDK's types, in one place: a body without the expected shape raises
``ResponseValidationError`` (an ``EndorError``, naming the endpoint and the first field that didn't fit), never
pydantic's ``ValidationError``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

from pydantic import BaseModel, TypeAdapter, ValidationError

from .errors import ResponseValidationError

__all__ = ["parse", "validated"]

M = TypeVar("M", bound=BaseModel)
T = TypeVar("T")


@overload
def parse(kind: type[M], raw: Any, endpoint: str | None, status: int = 200) -> M: ...


@overload
def parse(kind: TypeAdapter[T], raw: Any, endpoint: str | None, status: int = 200) -> T: ...


def parse(kind: type[BaseModel] | TypeAdapter[Any], raw: Any, endpoint: str | None, status: int = 200) -> Any:
    """``raw`` (a response body, or part of one) as ``kind``, a model or a ``TypeAdapter``."""
    if isinstance(kind, TypeAdapter):
        return validated(kind.validate_python, raw, endpoint, status)
    return validated(kind.model_validate, raw, endpoint, status)


def validated(validate: Callable[[Any], T], raw: Any, endpoint: str | None, status: int = 200) -> T:
    """``validate(raw)``, with a pydantic ``ValidationError`` raised as ``ResponseValidationError``."""
    try:
        return validate(raw)
    except ValidationError as e:
        first = e.errors(include_url=False)[0]
        path = ".".join(str(p) for p in first.get("loc", ())) or "<body>"
        raise ResponseValidationError(status, raw, None, f"{path}: {first.get('msg')}", endpoint) from e
