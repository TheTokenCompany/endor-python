"""Uploading images for training data (``client.images``): declare, PUT straight to storage, confirm.

The API never carries image bytes. ``POST /v1/images`` declares images by the SHA-256 of their bytes, their media
type and size, and answers, per image, ``ready`` (stored before) or ``upload`` with a link bound to that hash (storage
accepts only those bytes) and the headers to send. The SDK PUTs the files it has to (a few at once, with retries),
declares them again to confirm, and sends references, ``{"type": "image", "sha256"}``, in rows and datums. An image
is uploaded once per client (``_ready`` remembers the hashes) and stored once per organization.

Limited access: the API answers 403 ``image_training_not_enabled`` (``PermissionDeniedError``) until the organization
is approved for fine-tuning with images.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from . import _constants as C
from ._http import Transport
from ._log import logger
from .errors import EndorError
from .images import Image, as_image, find_images, map_images

__all__ = ["Images"]

DECLARE_BATCH = 1000  # images per POST /v1/images (the API's maximum)
UPLOAD_THREADS = 8  # files PUT at once
UPLOAD_TRIES = 3  # attempts per file: storage or the network failing
UPLOAD_TIMEOUT_S = 120.0  # one PUT (5 MB at most)


class Images:
    """``client.images``: images for training data, uploaded once and referenced by hash."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport
        self._ready: set[str] = set()  # sha256s this client confirmed uploaded
        self._lock = threading.Lock()

    def upload(self, images: Iterable[Image | dict[str, Any] | bytes | str | Path]) -> list[Image]:
        """Upload images for training data once and return their references, in order (``Image`` objects with
        ``sha256`` and no bytes). Each item is an ``Image``, a file's bytes, or a path to a PNG, JPEG or WebP file;
        a reference passes through. You rarely need this: ``project.datasets.upload``, ``run.forward``,
        ``run.forward_backward``, ``run.predict`` and ``client.predict`` upload the ``endor.Image`` objects in your
        rows and datums themselves.

        Raises:
            PermissionDeniedError: 403 ``image_training_not_enabled``: fine-tuning with images isn't enabled for the
                organization (limited access; ask us).
            LimitReachedError: the organization's stored bytes (datasets and images) are at their cap.
            EndorError: storage didn't take a file, or the API rejected an uploaded file.
        """
        items = [as_image(x) for x in images]
        self.ensure({im.sha256: im for im in items if not im.is_ref})
        return [Image.ref(im.sha256, im.media_type) for im in items]

    def prepare_states(self, holders: list[dict[str, Any]], key: str = "state") -> int:
        """Replace the inline images in ``holders[i][key]`` (rows' or datums' states) with references, uploading the
        images this client hasn't yet; how many images the states hold."""
        inline: dict[str, Image] = {}
        n = 0
        for h in holders:
            for _, im in find_images(h[key]):
                n += 1
                if "data" in im:
                    image = Image.wrap(im)
                    inline.setdefault(image.sha256, image)
        if inline:
            self.ensure(inline)
            for h in holders:
                h[key] = map_images(
                    h[key],
                    lambda im: dict(Image.ref(Image.wrap(im).sha256, im.get("media_type"))) if "data" in im else im,
                )
        return n

    def ensure(self, inline: dict[str, Image]) -> None:
        """Every one of these images (by sha256) uploaded and confirmed; those this client already has are skipped."""
        with self._lock:
            todo = [im for sha, im in inline.items() if sha not in self._ready]
            for i in range(0, len(todo), DECLARE_BATCH):
                batch = todo[i : i + DECLARE_BATCH]
                states = self._declare(batch)
                uploads = [(im, st) for im, st in zip(batch, states, strict=True) if st["status"] == "upload"]
                if uploads:
                    with ThreadPoolExecutor(min(UPLOAD_THREADS, len(uploads))) as pool:
                        list(pool.map(lambda x: self._put(*x), uploads))
                    states = self._declare(batch)  # confirms the uploads
                for st in states:
                    if st["status"] != "ready":
                        why = st.get("reason") or f"status {st['status']!r}"
                        raise EndorError(f"image {st['sha256'][:12]}... was not accepted: {why}")
                self._ready.update(im.sha256 for im in batch)

    def _declare(self, batch: list[Image]) -> list[dict[str, Any]]:
        body = {"images": [{"sha256": im.sha256, "media_type": im.media_type, "bytes": im.size} for im in batch]}
        r = self._t.request("POST", "/v1/images", json=body, method_name="client.images.upload")
        return list(r["images"])

    def _put(self, im: Image, st: dict[str, Any]) -> None:
        """PUT one file to its link with exactly the headers the API gave (the hash is signed: other bytes are
        refused). No Endor header goes to storage: the transport's plain client, an absolute URL."""
        data, err = im.content, ""
        for attempt in range(1, UPLOAD_TRIES + 1):
            try:
                r = self._t.client.put(st["url"], content=data, headers=dict(st["headers"]), timeout=UPLOAD_TIMEOUT_S)
                if r.status_code < 500:
                    if r.is_success:
                        return
                    raise EndorError(
                        f"storage refused image {im.sha256[:12]}...: {r.status_code} "
                        f"{r.text[: C.MAX_ERROR_BODY_LENGTH]}"
                    )
                err = f"{r.status_code}"
            except httpx.TransportError as e:
                err = type(e).__name__
            if attempt < UPLOAD_TRIES:
                logger.warning("uploading image %s...: %s; trying again", im.sha256[:12], err)
                time.sleep(attempt)
        raise EndorError(f"uploading image {im.sha256[:12]}... failed {UPLOAD_TRIES} times ({err})")
