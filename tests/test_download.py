"""project.models.download: files straight from storage, verified, links never kept, logged or sent the API key."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx
import pytest

import endor
from endor import DownloadError
from endor._download import fetch_all

from .fake_api import FakeEndor

BASE = ["adapter_model.safetensors", "adapter_config.json", "endor_manifest.json"]


@pytest.fixture
def saved(project: endor.Project) -> endor.Project:
    with project.runs.create("jev-9b") as run:
        run.save_checkpoint("v1", include_optimizer=True).result()
    return project


def test_writes_and_verifies_the_files(saved: endor.Project, fake: FakeEndor, tmp_path: Path) -> None:
    out = saved.models.download("v1", tmp_path / "v1")
    assert [f.name for f in out] == BASE  # optimizer.pt only on request
    assert sorted(p.name for p in (tmp_path / "v1").iterdir()) == sorted(BASE)
    for f in out:
        data = f.path.read_bytes()
        assert f.path == tmp_path / "v1" / f.name
        assert f.size_bytes == len(data) and f.sha256 == hashlib.sha256(data).hexdigest()
    (req,) = [r for r in fake.requests if r.path.endswith("/models/v1/download")]
    assert req.params == {"include_optimizer": "false"}
    with_opt = saved.models.download(f"{saved.name}/v1", tmp_path / "v1", include_optimizer=True)
    assert [f.name for f in with_opt] == [*BASE, "optimizer.pt"]


def test_storage_never_sees_the_api_key(saved: endor.Project, fake: FakeEndor, tmp_path: Path) -> None:
    saved.models.download("v1", tmp_path)
    assert len(fake.file_requests) == 3
    for r in fake.file_requests:
        assert "authorization" not in r.headers and not any(h.startswith("x-endor") for h in r.headers)


def test_links_are_never_logged_or_returned(
    saved: endor.Project, fake: FakeEndor, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    out = saved.models.download("v1", tmp_path)
    tokens = list(fake.file_links)
    text = caplog.text + repr(out) + str([f.model_dump() for f in out])
    assert "X-Amz-Signature" not in text and not any(t in text for t in tokens)
    assert "files.endor.test" in caplog.text  # httpx did log the requests, scrubbed


def test_a_corrupt_file_raises_and_leaves_nothing(saved: endor.Project, fake: FakeEndor, tmp_path: Path) -> None:
    fake.corrupt_files = True
    with pytest.raises(DownloadError, match="SHA-256 mismatch") as e:
        saved.models.download("v1", tmp_path)
    assert "X-Amz" not in str(e.value) and list(tmp_path.iterdir()) == []


def test_expired_links_and_transport_errors_never_show_the_link(tmp_path: Path) -> None:
    url = "https://files.endor.test/t/adapter_config.json?X-Amz-Signature=secret"
    expired = httpx.MockTransport(lambda r: httpx.Response(403))
    with pytest.raises(DownloadError, match="expired") as e:
        fetch_all([{"name": "adapter_config.json", "size_bytes": 1, "url": url}], tmp_path, transport=expired)
    assert "secret" not in str(e.value)

    def boom(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {r.url}")

    with pytest.raises(DownloadError) as e:
        fetch_all(
            [{"name": "adapter_config.json", "size_bytes": 1, "url": url}],
            tmp_path,
            transport=httpx.MockTransport(boom),
        )
    assert "secret" not in str(e.value) and e.value.__cause__ is None and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../evil", "/etc/passwd", ".bashrc", "sub/adapter_config.json", None, "x.py"])
def test_only_known_file_names_are_written(tmp_path: Path, name: object) -> None:
    ok = httpx.MockTransport(lambda r: httpx.Response(200, content=b"x"))
    with pytest.raises(DownloadError, match="unexpected file name"):
        fetch_all([{"name": name, "size_bytes": 1, "url": "https://s/x"}], tmp_path / "out", transport=ok)
    assert [p.name for p in tmp_path.iterdir()] == ["out"] and list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("url", ["http://s/x", "file:///etc/passwd", "", None])
def test_only_https_links(tmp_path: Path, url: object) -> None:
    with pytest.raises(DownloadError, match="not HTTPS"):
        fetch_all([{"name": "adapter_config.json", "size_bytes": 1, "url": url}], tmp_path)


def test_redirects_are_not_followed(tmp_path: Path) -> None:
    hop = httpx.MockTransport(
        lambda r: (
            httpx.Response(302, headers={"location": "https://elsewhere/x"})
            if r.url.host == "s"
            else httpx.Response(200, content=b"x")
        )
    )
    with pytest.raises(DownloadError, match="HTTP 302"):
        fetch_all([{"name": "adapter_config.json", "size_bytes": 1, "url": "https://s/x"}], tmp_path, transport=hop)


def test_size_mismatch(tmp_path: Path) -> None:
    ok = httpx.MockTransport(lambda r: httpx.Response(200, content=b"xx"))
    with pytest.raises(DownloadError, match="got 2 bytes, expected 3"):
        fetch_all([{"name": "adapter_config.json", "size_bytes": 3, "url": "https://s/x"}], tmp_path, transport=ok)
    assert list(tmp_path.iterdir()) == []
