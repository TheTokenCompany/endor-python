"""Downloading files from the short-lived links the API hands out: a saved model's files
(``client.custom_models.download``, ``project.download``) and an evaluation's results (``client.predict``,
``fetch_bytes``).

The links are presigned storage URLs: anyone holding one can read that file until it expires, so they never leave
this module. They are not returned, not put in exceptions and not logged (``httpx`` logs every request URL at INFO,
so its log records are scrubbed while the SDK is imported). The API key is never sent to storage: each file is fetched
with a separate HTTP client that has no Endor headers, over HTTPS only, without following redirects.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

from .errors import DownloadError
from .types import DownloadedFile

# The only names the SDK writes, whatever the server says: no paths, no hidden files.
OPTIMIZER = "optimizer.pt"
ALLOWED = frozenset(
    {"adapter_model.safetensors", "adapter_config.json", "endor_manifest.json", "readout.safetensors", OPTIMIZER}
)
_SIGNED = re.compile(r"(https?://[^/\s\"']+)[^\s\"']*(?:X-Amz-|Signature=)[^\s\"']*")
CHUNK = 1 << 20


class _ScrubLinks(logging.Filter):
    """Keeps only the host of any signed URL in a log record of httpx / httpcore."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "X-Amz-" in msg or "Signature=" in msg:
            record.msg, record.args = _SIGNED.sub(r"\1/[redacted]", msg), None
        return True


def _install_log_scrubber() -> None:
    for name in ("httpx", "httpcore"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _ScrubLinks) for f in lg.filters):
            lg.addFilter(_ScrubLinks())


_install_log_scrubber()


def fetch_all(
    files: list[dict[str, Any]],
    folder: str | os.PathLike[str],
    *,
    what: str = "the model",
    include_optimizer: bool = False,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 60.0,
) -> list[DownloadedFile]:
    """Write each file to ``folder`` (created if needed) and verify it; a file is only in place once whole.
    ``include_optimizer``: optimizer.pt was asked for, so a model saved without optimizer state (the API then lists no
    optimizer.pt) raises ``DownloadError`` naming ``what`` before anything is written."""
    if include_optimizer and not any(f.get("name") == OPTIMIZER for f in files):
        raise DownloadError(
            f"{what} was saved without optimizer state, so it has no {OPTIMIZER}: download it without "
            "include_optimizer. Only run.save_checkpoint(include_optimizer=True) stores optimizer state."
        )
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    done: list[DownloadedFile] = []
    with httpx.Client(transport=transport, timeout=timeout, follow_redirects=False) as http:
        for f in files:
            name = f.get("name")
            if name not in ALLOWED:
                raise DownloadError(f"the API offered an unexpected file name {name!r}; nothing was written for it")
            done.append(_fetch(http, f, out / name))
    return done


def _fetch(http: httpx.Client, f: dict[str, Any], dest: Path) -> DownloadedFile:
    name, url = f["name"], f.get("url") or ""
    if not url.startswith("https://"):
        raise DownloadError(f"{name}: the API returned a link that is not HTTPS; refusing to download it")
    expected_size, expected_sha = f.get("size_bytes"), f.get("sha256")
    part = dest.with_name(dest.name + ".part")
    h, size = hashlib.sha256(), 0
    try:
        with http.stream("GET", url) as r:
            if r.status_code != 200:
                hint = " (the link expired; download again)" if r.status_code == 403 else ""
                raise DownloadError(f"{name}: storage answered HTTP {r.status_code}{hint}")
            with part.open("wb") as fh:
                for chunk in r.iter_bytes(CHUNK):
                    fh.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
        if expected_size is not None and size != expected_size:
            raise DownloadError(f"{name}: got {size} bytes, expected {expected_size}")
        digest = h.hexdigest()
        if expected_sha and digest != expected_sha.lower():
            raise DownloadError(f"{name}: SHA-256 mismatch (got {digest}, expected {expected_sha})")
        os.replace(part, dest)
    except httpx.HTTPError as e:  # its message would carry the link
        part.unlink(missing_ok=True)
        raise DownloadError(f"{name}: download failed ({type(e).__name__})") from None
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return DownloadedFile(name=name, path=dest, size_bytes=size, sha256=digest)


def fetch_bytes(url: str, what: str, *, transport: httpx.BaseTransport | None = None, timeout: float = 60.0) -> bytes:
    """One file's bytes from a presigned link (an evaluation's results). ``what`` names it in errors."""
    _check_link(url, what)
    try:
        with httpx.Client(transport=transport, timeout=timeout, follow_redirects=False) as http:
            r = http.get(url)
    except httpx.HTTPError as e:  # its message would carry the link
        raise DownloadError(f"{what}: download failed ({type(e).__name__})") from None
    return _body(r, what)


async def fetch_bytes_async(
    url: str, what: str, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 60.0
) -> bytes:
    """``fetch_bytes`` for async code."""
    _check_link(url, what)
    try:
        async with httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False) as http:
            r = await http.get(url)
    except httpx.HTTPError as e:  # its message would carry the link
        raise DownloadError(f"{what}: download failed ({type(e).__name__})") from None
    return _body(r, what)


def _check_link(url: str, what: str) -> None:
    if not url.startswith("https://"):
        raise DownloadError(f"{what}: the API returned a link that is not HTTPS; refusing to download it")


def _body(r: httpx.Response, what: str) -> bytes:
    if r.status_code != 200:
        hint = " (the link expired; fetch it again)" if r.status_code == 403 else ""
        raise DownloadError(f"{what}: storage answered HTTP {r.status_code}{hint}")
    return r.content
