"""endor.Image: the wire object for an image in a state, built from a file, bytes or a Pillow image, sent as is."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

import endor
from endor import EndorClient, Image, UnprocessableEntityError

from .conftest import URGENT, unique
from .fake_api import FakeEndor

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x01" * 32
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x02" * 32


def test_from_bytes_and_path(tmp_path: Path) -> None:
    im = Image.from_bytes(PNG)
    assert im == {"type": "image", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}
    assert isinstance(im, dict) and im.media_type == "image/png" and im.size == len(PNG)
    assert repr(im) == f"Image(image/png, {len(PNG)} bytes)"
    assert Image.from_bytes(JPEG).media_type == "image/jpeg" and Image.from_bytes(WEBP).media_type == "image/webp"
    assert Image.from_bytes(b"anything", "image/webp")["media_type"] == "image/webp"  # given: not checked here
    (tmp_path / "a.jpg").write_bytes(JPEG)
    assert Image.from_path(tmp_path / "a.jpg") == Image.from_bytes(JPEG)
    with pytest.raises(ValueError, match="PNG, JPEG or WebP"):
        Image.from_bytes(b"GIF89a....")
    with pytest.raises(ValueError, match="media_type"):
        Image.from_bytes(PNG, "image/gif")


def test_from_pil() -> None:
    pil = pytest.importorskip("PIL.Image")
    im = Image.from_pil(pil.new("RGBA", (4, 3), (255, 0, 0, 128)))
    assert im.media_type == "image/png" and base64.b64decode(im["data"]).startswith(b"\x89PNG")
    jpg = Image.from_pil(pil.new("RGBA", (4, 3)), format="jpg")  # RGBA can't be JPEG: converted to RGB
    assert jpg.media_type == "image/jpeg"
    with pytest.raises(ValueError, match="format"):
        Image.from_pil(pil.new("RGB", (1, 1)), format="GIF")


def test_images_go_anywhere_in_the_state(client: EndorClient, fake: FakeEndor) -> None:
    photos = client.projects.create(unique("photos"), base_models=["gev-26b"]).name
    state = {"photo": Image.from_bytes(PNG), "more": ["caption", Image.from_bytes(JPEG)]}
    res = client.system_one(state, {"u": URGENT}, model=photos)
    sent = fake.requests[-1].body["state"]
    assert sent == {"photo": dict(Image.from_bytes(PNG)), "more": ["caption", dict(Image.from_bytes(JPEG))]}
    assert res.usage.input_tokens > 2 * 280  # the fake counts 280 tokens an image
    top = client.system_one(Image.from_bytes(WEBP), {"u": URGENT}, model=photos)  # the whole state an image
    assert fake.requests[-1].body["state"]["media_type"] == "image/webp" and top.nouls["u"].noul == 0.5


async def test_async_and_text_only_bases(client: EndorClient, decider: str) -> None:
    with pytest.raises(UnprocessableEntityError) as e:
        client.system_one({"photo": Image.from_bytes(PNG)}, {"u": URGENT}, model=decider)  # jev-9b reads text only
    assert e.value.code == "unsupported_modality" and e.value.param == "state.photo"
    photos = client.projects.create(unique("photos"), base_models=["gev-26b"]).name
    res = await client.system_one_async([Image.from_bytes(PNG)], {"u": URGENT}, model=photos)
    assert res.model == f"{photos}/gev-26b"


def test_base_models_say_which_read_images(client: EndorClient) -> None:
    bases = {b.id: b.modalities for b in client.base_models()}
    assert bases["gev-26b"] == ["text", "image"] and bases["jev-9b"] == ["text"]
    assert endor.Image is Image
