"""endor.Image: the wire object for an image in a state, built from a file, bytes or a Pillow image, sent as is."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import httpx
import pytest

import endor
from endor import Datum, EndorClient, Image, Target, UnprocessableEntityError, images

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


# ---------------------------------------------------------------- images in training data


def png(n: int) -> bytes:
    """A distinct PNG-looking file (the fake checks the first bytes, as the API does)."""
    return b"\x89PNG\r\n\x1a\n" + n.to_bytes(4, "big") * 16


def test_a_reference_names_an_uploaded_image_by_hash() -> None:
    im = Image.from_bytes(png(1))
    assert not im.is_ref and im.sha256 == hashlib.sha256(png(1)).hexdigest() and im.content == png(1)
    ref = Image.ref(im.sha256, "image/png")
    assert ref == {"type": "image", "sha256": im.sha256, "media_type": "image/png"} and ref.is_ref
    assert ref.sha256 == im.sha256 and ref.size == 0 and repr(ref).startswith("Image(ref ")
    assert Image.ref(im.sha256) == {"type": "image", "sha256": im.sha256}
    with pytest.raises(ValueError, match="no bytes"):
        _ = ref.content
    with pytest.raises(ValueError, match="hex"):
        Image.ref("not-a-hash")
    assert Image.wrap(dict(ref)).is_ref and Image.wrap(dict(im)) == im
    assert images.find_images({"a": im, "b": [1, ref], "c": {"type": "image", "alt": "x"}}) == [
        ("state.a", im),
        ("state.b.1", ref),
    ]


def test_images_are_declared_uploaded_once_and_confirmed(client: EndorClient, fake: FakeEndor, tmp_path: Path) -> None:
    """client.images.upload: one declaration, a PUT per missing image straight to the link with exactly the headers
    the API gave (and no Endor header), a second declaration to confirm; references back in order; an image this
    client uploaded is never declared again."""
    (tmp_path / "c.png").write_bytes(png(3))
    refs = client.images.upload([Image.from_bytes(png(1)), png(2), tmp_path / "c.png", Image.from_bytes(png(1))])
    shas = [hashlib.sha256(png(i)).hexdigest() for i in (1, 2, 3)]
    assert [r.sha256 for r in refs] == [shas[0], shas[1], shas[2], shas[0]] and all(r.is_ref for r in refs)
    assert refs[0].media_type == "image/png"
    declared = [r for r in fake.requests if r.path == "/v1/images"]
    assert len(declared) == 2 and [x["sha256"] for x in declared[0].body["images"]] == shas  # once per image
    assert declared[0].body["images"][0] == {"sha256": shas[0], "media_type": "image/png", "bytes": len(png(1))}
    puts = fake.upload_requests
    assert sorted(r.path.strip("/") for r in puts) == sorted(shas) and all(r.method == "PUT" for r in puts)
    assert all("authorization" not in r.headers and "x-endor-method" not in r.headers for r in puts)
    assert {sha: row["status"] for sha, row in fake.images.items()} == dict.fromkeys(shas, "ready")
    assert fake.uploads[shas[1]] == png(2)
    before = len(fake.requests)
    again = client.images.upload([png(2), Image.ref(shas[0])])  # known: nothing declared, nothing uploaded
    assert [r.sha256 for r in again] == [shas[1], shas[0]] and len(fake.requests) == before
    # another client knows nothing yet: it declares, finds them ready, uploads nothing
    other = endor.EndorClient(api_key="edk_test", base_url=client.base_url, transport=httpx.MockTransport(fake.handler))
    with other:
        other.images.upload([png(1)])
    assert len(fake.upload_requests) == 3 and fake.requests[-1].path == "/v1/images"
    listed = client._t.request("GET", "/v1/images")
    assert listed["total"] == 3


def test_rows_and_datums_carry_images_as_references(client: EndorClient, fake: FakeEndor) -> None:
    photos = client.projects.create(unique("photos"), base_models=["gev-26b"])
    rows = [
        {
            "id": f"r{i}",
            "state": {"photo": Image.from_bytes(png(i % 3)), "shelf": f"s{i}"},
            "questions": {"u": URGENT},
            "labels": {"u": i % 2 == 0},
        }
        for i in range(6)
    ]
    info = photos.datasets.upload("imgs", rows)
    assert info.n_images == 6 and info.n_rows == 6
    sent = [r for r in fake.requests if r.path.endswith("/datasets")][-1].body["rows"]
    assert all("data" not in r["state"]["photo"] and r["state"]["photo"]["sha256"] for r in sent)
    assert len({r["state"]["photo"]["sha256"] for r in sent}) == 3 and len(fake.uploads) == 3
    assert sent[0]["state"]["shelf"] == "s0" and sent[0]["labels"] == {"u": True}
    # training: datums with images, uploaded once, sent as references, split by images as well as by datums
    with photos.runs.create(base_model="gev-26b") as run:
        datums = [
            Datum(state={"photo": Image.from_bytes(png(i)), "t": "x"}, question=URGENT, target=Target(label="true"))
            for i in range(300)
        ]
        out = run.forward_backward(datums).result()
        assert len(out.probabilities) == 300
        calls = [r for r in fake.requests if r.path.endswith("/forward_backward")]
        assert [len(c.body["data"]) for c in calls] == [128, 128, 44]  # 128 images a call
        assert all("data" not in d["state"]["photo"] for c in calls for d in c.body["data"])
        assert len(fake.uploads) == 300  # 0..2 were there already
        answers = run.predict([{"state": [Image.from_bytes(png(1)), "left"], "questions": {"u": URGENT}}]).result()
        assert answers[0]["u"].noul == 0.5
        sent_rows = [r for r in fake.requests if r.path.endswith("/predict")][-1].body["rows"]
        assert sent_rows[0]["state"][0] == {
            "type": "image",
            "sha256": hashlib.sha256(png(1)).hexdigest(),
            "media_type": "image/png",
        }
        run.optim_step(learning_rate=1e-4).result()
        run.save_checkpoint("v1").result()
    answers = client.predict(
        f"{photos.name}/v1", [{"state": {"photo": Image.from_bytes(png(7))}, "questions": {"u": URGENT}}]
    )
    assert len(answers) == 1
    ev = [r for r in fake.requests if r.path.endswith("/evaluations")][-1].body
    assert "data" not in ev["rows"][0]["state"]["photo"] and hashlib.sha256(png(7)).hexdigest() in fake.uploads


def test_without_access_the_error_says_how_to_ask(client: EndorClient, fake: FakeEndor, decider: str) -> None:
    fake.image_training = False
    assert client.whoami().image_training is False
    with pytest.raises(endor.PermissionDeniedError) as e:
        client.images.upload([png(1)])
    assert e.value.code == "image_training_not_enabled" and "request access" in e.value.message
    photos = client.projects.create(unique("photos"), base_models=["gev-26b"])
    with pytest.raises(endor.PermissionDeniedError):
        photos.datasets.upload("imgs", [{"state": [Image.from_bytes(png(1))], "questions": {"u": URGENT}}])
    assert not fake.uploads and not fake.images  # nothing was uploaded
    fake.image_training = True
    assert client.whoami().image_training is True
    # a base that reads text only refuses image datums, after the upload (the API's order)
    tickets = client.projects.get(decider.partition("/")[0])
    with tickets.runs.create() as run:
        with pytest.raises(UnprocessableEntityError) as err:
            run.forward_backward(
                [Datum(state=[Image.from_bytes(png(2))], question=URGENT, target=Target(label="true"))]
            )
        assert err.value.code == "unsupported_modality"


def test_an_upload_that_storage_keeps_refusing_raises(client: EndorClient, fake: FakeEndor, monkeypatch) -> None:
    from endor import _images

    monkeypatch.setattr(_images.time, "sleep", lambda s: None)
    fake.fail_uploads = 2  # two 500s, then it goes through
    assert client.images.upload([png(1)])[0].is_ref and len(fake.upload_requests) == 3
    fake.fail_uploads = 5
    with pytest.raises(endor.EndorError, match="failed 3 times"):
        client.images.upload([png(2)])
    # a file the API rejects after the upload (here: a declared PNG whose bytes aren't one)
    fake.fail_uploads = 0
    jpeg_as_png = Image(b"\xff\xd8\xff\xe0" + b"\x01" * 32, "image/png")
    with pytest.raises(endor.EndorError, match="not accepted"):
        client.images.upload([jpeg_as_png])
