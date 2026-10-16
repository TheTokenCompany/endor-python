"""Images in a state: a decision's, or training data's.

An image is the JSON object ``{"type": "image", "media_type": "image/png", "data": "<base64>"}`` anywhere a state value
can be: the whole state, an item of a list, a value of an object. Only base models that read images take them
(``BaseModelInfo.modalities`` includes ``"image"``); others answer ``422 unsupported_modality``.

    state = {"screenshot": endor.Image.from_path("checkout.png"), "goal": "Pay for the order"}
    client.system_one(state, questions, model="shop")

The same object goes in training data: dataset rows, ``forward_backward`` datums and ``predict`` rows. There the SDK
uploads each image once (``client.images``) and sends a reference, ``{"type": "image", "sha256": "<hex>"}``, which
``Image.ref`` builds by hand. Fine-tuning with images is in limited access (``PermissionDeniedError``, 403
``image_training_not_enabled``, until the organization is approved; ``client.whoami().image_training``).
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["Image", "MEDIA_TYPES", "find_images", "map_images", "as_image"]

MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp")
_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}  # Pillow's format names
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class Image(dict):  # type: ignore[type-arg]
    """An image for a state. It is the wire object itself (a ``dict``), so it serializes as is and can go anywhere in
    a state. Build it with ``from_path``, ``from_bytes`` or ``from_pil``; ``ref`` builds a reference to an image
    already uploaded for training data (``sha256``, no bytes)."""

    def __init__(self, data: bytes | None = None, media_type: str | None = None, *, sha256: str | None = None) -> None:
        if data is None and sha256 is None:
            raise ValueError("an Image needs its bytes (data), or the sha256 of an uploaded image")
        if media_type is not None and media_type not in MEDIA_TYPES:
            raise ValueError(f"media_type must be one of {', '.join(MEDIA_TYPES)}, not {media_type!r}")
        if data is not None:
            if media_type is None:
                raise ValueError("an image's bytes need their media_type")
            super().__init__(type="image", media_type=media_type, data=base64.b64encode(data).decode("ascii"))
            self._sha256: str | None = hashlib.sha256(data).hexdigest()
            return
        if not isinstance(sha256, str) or not _SHA256.match(sha256):
            raise ValueError("sha256 is the SHA-256 of the image's bytes, 64 lowercase hex characters")
        fields: dict[str, Any] = {"type": "image", "sha256": sha256}
        if media_type is not None:
            fields["media_type"] = media_type
        super().__init__(**fields)
        self._sha256 = sha256

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str | None = None) -> Image:
        """An image from a file's bytes. Without ``media_type``, it is read from the first bytes (PNG, JPEG or
        WebP)."""
        media_type = media_type or _sniff(data)
        if media_type is None:
            raise ValueError("not a PNG, JPEG or WebP image; give media_type, or convert it to one of these")
        return cls(data, media_type)

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        """An image from a PNG, JPEG or WebP file."""
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_pil(cls, image: Any, format: str = "PNG") -> Image:  # noqa: A002 (Pillow's name)
        """An image from a Pillow image, saved as ``format`` (PNG, JPEG or WebP). Needs Pillow, which the SDK does not
        install."""
        fmt = format.upper().replace("JPG", "JPEG")
        if fmt not in _FORMATS:
            raise ValueError(f"format must be PNG, JPEG or WEBP, not {format!r}")
        if fmt == "JPEG" and image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return cls(buf.getvalue(), _FORMATS[fmt])

    @classmethod
    def ref(cls, sha256: str, media_type: str | None = None) -> Image:
        """A reference to an image uploaded for training data (``client.images.upload``), by the SHA-256 of its
        bytes. Datasets, datums and predict rows take it; a decision needs the bytes."""
        return cls(None, media_type, sha256=sha256)

    @classmethod
    def wrap(cls, wire: dict[str, Any]) -> Image:
        """An ``Image`` from the wire object (a dict with ``data`` or ``sha256``), as the API or a row file has it."""
        if isinstance(wire, Image):
            return wire
        if "data" in wire:
            return cls(base64.b64decode(wire["data"]), wire.get("media_type"))
        return cls(None, wire.get("media_type"), sha256=wire.get("sha256"))

    @property
    def is_ref(self) -> bool:
        """A reference to an uploaded image (no bytes here)."""
        return "data" not in self

    @property
    def media_type(self) -> str | None:
        v = self.get("media_type")
        return None if v is None else str(v)

    @property
    def sha256(self) -> str:
        """The SHA-256 of the image's bytes, in hex: the name an uploaded image has."""
        sha = getattr(self, "_sha256", None)
        if sha is None:
            sha = self._sha256 = str(self["sha256"]) if self.is_ref else hashlib.sha256(self.content).hexdigest()
        return sha

    @property
    def content(self) -> bytes:
        """The image's bytes (decoded). A reference has none: ``ValueError``."""
        if self.is_ref:
            raise ValueError(f"image {self.sha256[:12]}... is a reference to an uploaded image: it has no bytes here")
        return base64.b64decode(str(self["data"]))

    @property
    def size(self) -> int:
        """The image's size in bytes (before base64); 0 for a reference."""
        if self.is_ref:
            return 0
        data = str(self["data"])
        return len(data) * 3 // 4 - data[-2:].count("=")

    def __repr__(self) -> str:
        if self.is_ref:
            return f"Image(ref {self.sha256[:12]}...)"
        return f"Image({self.media_type}, {self.size} bytes)"


def as_image(x: Image | dict[str, Any] | bytes | str | Path) -> Image:
    """An ``Image`` from what ``client.images.upload`` takes: an ``Image`` or its wire dict, a file's bytes, or a
    path."""
    if isinstance(x, Image):
        return x
    if isinstance(x, dict):
        return Image.wrap(x)
    if isinstance(x, (bytes, bytearray, memoryview)):
        return Image.from_bytes(bytes(x))
    return Image.from_path(x)


def is_image(v: Any) -> bool:
    return isinstance(v, dict) and v.get("type") == "image" and ("data" in v or "sha256" in v)


def find_images(v: Any, path: str = "state") -> list[tuple[str, dict[str, Any]]]:
    """Every image in a state with its path (``state``, ``state.photo``, ``state.2``), in reading order: object values
    in key order, list items in order, depth first."""
    if is_image(v):
        return [(path, v)]
    if isinstance(v, dict):
        return [x for k, item in v.items() for x in find_images(item, f"{path}.{k}")]
    if isinstance(v, list):
        return [x for i, item in enumerate(v) for x in find_images(item, f"{path}.{i}")]
    return []


def map_images(v: Any, f: Callable[[dict[str, Any]], Any]) -> Any:
    """The state with each image replaced by ``f(image)``; everything else as it was."""
    if is_image(v):
        return f(v)
    if isinstance(v, dict):
        return {k: map_images(item, f) for k, item in v.items()}
    if isinstance(v, list):
        return [map_images(item, f) for item in v]
    return v
