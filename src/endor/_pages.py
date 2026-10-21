"""Paged lists: every item of a ``{"items", "total"}`` list, or at most ``limit`` of them."""

from __future__ import annotations

from typing import Any

from . import _constants as C
from ._http import Transport


def list_all(
    t: Transport,
    path: str,
    params: dict[str, Any],
    method_name: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[Any]:
    """Every item of a paged list (``{"items", "total"}``), or at most ``limit`` of them, starting at ``offset``."""
    out: list[Any] = []
    while limit is None or len(out) < limit:
        size = C.LIST_PAGE_SIZE if limit is None else min(C.LIST_PAGE_SIZE, limit - len(out))
        r = t.request("GET", path, params={**params, "limit": size, "offset": offset}, method_name=method_name)
        items = r["items"]
        out += items
        offset += len(items)
        if not items or offset >= r.get("total", offset):
            break
    return out
