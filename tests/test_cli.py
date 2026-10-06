from __future__ import annotations

import json
from pathlib import Path

import pytest

import endor
from endor import EndorClient, cli

from .conftest import rows, unique
from .fake_api import FakeEndor


def run(capsys: pytest.CaptureFixture[str], client: EndorClient, *args: str) -> str:
    assert cli.main(list(args), client=client) == 0
    return capsys.readouterr().out


def run_json(capsys: pytest.CaptureFixture[str], client: EndorClient, *args: str) -> object:
    return json.loads(run(capsys, client, "-f", "json", *args))


def test_whoami_and_base_models(capsys: pytest.CaptureFixture[str], client: EndorClient) -> None:
    assert "user_id=usr_test" in run(capsys, client, "whoami")
    bases = run_json(capsys, client, "base-models")
    assert isinstance(bases, list) and {b["id"] for b in bases} >= {"jevk5-4b"}


def test_projects_datasets_runs_models(capsys: pytest.CaptureFixture[str], client: EndorClient, tmp_path: Path) -> None:
    name = unique("cli")
    assert run_json(capsys, client, "projects", "create", name)["name"] == name  # type: ignore[index]
    assert name in [p["name"] for p in run_json(capsys, client, "projects", "list")]  # type: ignore[union-attr]
    f = tmp_path / "rows.jsonl"
    endor.data.save_rows(f, rows(4))
    ds = run_json(capsys, client, "datasets", "upload", name, "train", str(f))
    assert ds["n_rows"] == 4  # type: ignore[index]
    assert [d["name"] for d in run_json(capsys, client, "datasets", "list", name)] == ["train"]  # type: ignore[union-attr]
    project = client.projects.get(name)
    r = project.runs.create("jevk5-4b", name="cli-run")
    r.save_checkpoint("v1").result()
    assert "cli-run" in run(capsys, client, "runs", "list", name)
    assert run_json(capsys, client, "runs", "show", r.id)["name"] == "cli-run"  # type: ignore[index]
    assert run_json(capsys, client, "models", "info", f"{name}/v1")["step"] == 0  # type: ignore[index]
    assert run_json(capsys, client, "models", "set-ttl", f"{name}/v1", "7200")["expires_at"]  # type: ignore[index]
    assert run_json(capsys, client, "models", "set-ttl", f"{name}/v1", "none")["expires_at"] is None  # type: ignore[index]
    ev = run_json(capsys, client, "eval", name, f"{name}/v1", "train")
    assert ev["status"] == "completed"  # type: ignore[index]
    assert run_json(capsys, client, "runs", "close", r.id)["status"] == "closed"  # type: ignore[index]
    run(capsys, client, "models", "delete", f"{name}/v1")
    assert run_json(capsys, client, "models", "list", name) == []
    run(capsys, client, "datasets", "delete", name, "train")
    run(capsys, client, "projects", "delete", name)
    assert name not in [p["name"] for p in run_json(capsys, client, "projects", "list")]  # type: ignore[union-attr]


def test_usage(capsys: pytest.CaptureFixture[str], client: EndorClient) -> None:
    out = run(capsys, client, "usage", "--start", "2026-10-01", "--end", "2026-10-03", "--csv")
    assert out.splitlines()[0] == "hour,kind,project,base_model,run_id,tokens,cost_usd"
    assert run_json(capsys, client, "usage", "--start", "2026-10-01", "--end", "2026-10-03") == []


def test_errors_exit_nonzero(capsys: pytest.CaptureFixture[str], client: EndorClient) -> None:
    assert cli.main(["projects", "create", "Bad Name"], client=client) == 1
    assert "error: POST /v1/projects: 422 invalid_input" in capsys.readouterr().err
    assert cli.main(["models", "info", "no-slash"], client=client) == 1
    assert "<project>/<name>" in capsys.readouterr().err


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and endor.__version__ in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main([])


def test_download(
    capsys: pytest.CaptureFixture[str],
    client: EndorClient,
    fake: FakeEndor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    project = client.projects.create(unique("dl"))
    project.runs.create("jevk5-4b").save_checkpoint("v1").result()

    class FakeStream:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self) -> list[bytes]:
            return [b"tar", b"bytes"]

    monkeypatch.setattr(httpx, "stream", FakeStream)
    target = tmp_path / "m.tar"
    out = run(capsys, client, "models", "download", f"{project.name}/v1", "-o", str(target))
    assert out.strip() == str(target) and target.read_bytes() == b"tarbytes"
