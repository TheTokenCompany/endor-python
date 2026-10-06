"""Request headers that identify the SDK and the SDK method behind each request.

Every request carries (see docs/REQUEST_HEADERS.md):

    User-Agent:                endor-python/<version> (python 3.12.1; darwin; arm64)   "; cli" added for the CLI
    X-Endor-SDK-Method:        <the public SDK method, e.g. "run.forward_backward">
    X-Endor-SDK-Recipe:        <recipe name, only inside a recipe, e.g. "supervised">
    X-Endor-Client-Request-Id: <uuid4, the same across retries of one logical call>
    X-Endor-Retry-Count:       <n, only on retries>
    Idempotency-Key:           <uuid4, on resource-creating POSTs>
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from ._constants import HEADER_SDK_METHOD, HEADER_SDK_RECIPE, SDK_NAME
from ._version import __version__

RUNTIME = f"python {platform.python_version()}; {sys.platform}; {platform.machine() or 'unknown'}"

_interface: ContextVar[str] = ContextVar("endor_interface", default="python")
_recipe: ContextVar[str | None] = ContextVar("endor_recipe", default=None)


def user_agent() -> str:
    """``endor-python/<version> (<runtime>)``, with ``; cli`` when the request comes from the ``endor`` command."""
    cli = "; cli" if _interface.get() == "cli" else ""
    return f"{SDK_NAME}/{__version__} ({RUNTIME}{cli})"


def identity_headers() -> dict[str, str]:
    """The headers that say which SDK, version and runtime a request comes from."""
    return {"User-Agent": user_agent()}


def context_headers(method: str | None) -> dict[str, str]:
    """Per-call headers: the SDK method and the active recipe, if any."""
    out: dict[str, str] = {}
    if method:
        out[HEADER_SDK_METHOD] = method
    recipe = _recipe.get()
    if recipe:
        out[HEADER_SDK_RECIPE] = recipe
    return out


@contextmanager
def sdk_context(*, interface: str | None = None, recipe: str | None = None) -> Iterator[None]:
    """Tag every request made inside the block (used by the CLI and the recipes)."""
    tokens: list[tuple[ContextVar[Any], Token[Any]]] = []
    if interface is not None:
        tokens.append((_interface, _interface.set(interface)))
    if recipe is not None:
        tokens.append((_recipe, _recipe.set(recipe)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
