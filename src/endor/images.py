"""Images in a decision's state.

An image is the JSON object ``{"type": "image", "media_type": "image/png", "data": "<base64>"}`` anywhere a state value
can be: the whole state, an item of a list, a value of an object. Only base models that read images take them
(``BaseModelInfo.modalities`` includes ``"image"``); others answer ``422 unsupported_modality``.

    state = {"screenshot": endor.Image.from_path("checkout.png"), "goal": "Pay for the order"}
    client.system_one(state, questions, model="shop")
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

__all__ = ["Image", "MEDIA_TYPES"]

MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp")
_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}  # Pillow's format names


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class Image(dict):  # type: ignore[type-arg]
    """An image for a decision's state. It is the wire object itself (a ``dict``), so it serializes as is and can go
    anywhere in a state. Build it with ``from_path``, ``from_bytes`` or ``from_pil``."""

    def __init__(self, data: bytes, media_type: str) -> None:
        if media_type not in MEDIA_TYPES:
            raise ValueError(f"media_type must be one of {', '.join(MEDIA_TYPES)}, not {media_type!r}")
        super().__init__(type="image", media_type=media_type, data=base64.b64encode(data).decode("ascii"))

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

    @property
    def media_type(self) -> str:
        return str(self["media_type"])

    @property
    def size(self) -> int:
        """The image's size in bytes (before base64)."""
        data = str(self["data"])
        return len(data) * 3 // 4 - data[-2:].count("=")

    def __repr__(self) -> str:
        return f"Image({self.media_type}, {self.size} bytes)"
