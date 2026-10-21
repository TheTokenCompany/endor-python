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
    out = run(capsys, client, "whoami")
    assert "org_id=" in out and "balance_usd=" in out
    me = run_json(capsys, client, "whoami")
    assert isinstance(me, dict) and me["org_id"] and "limits" in me and "balance_usd" in me
    bases = run_json(capsys, client, "base-models")
    assert isinstance(bases, list) and {b["id"] for b in bases} >= {"jev-9b"}


def test_projects(capsys: pytest.CaptureFixture[str], client: EndorClient, fake: FakeEndor) -> None:
    name = unique("cli")
    created = run_json(capsys, client, "projects", "create", name, "--base-model", "jev-9b")
    assert created["name"] == name and created["base_model"] == "jev-9b"  # type: ignore[index]
    assert name in [p["name"] for p in run_json(capsys, client, "projects", "list")]  # type: ignore[union-attr]
    fake.add_version(name)
    fake.add_version(name)  # auto-promote is off by default: v2 is the preview, nothing live yet
    versions = run_json(capsys, client, "projects", "versions", name)
    assert [(v["name"], v["production"], v["preview"]) for v in versions] == [  # type: ignore[index,union-attr]
        ("v2", False, True),
        ("v1", False, False),
    ]
    promoted = run_json(capsys, client, "projects", "promote", name, "v1")
    assert promoted["production"] == "v1" and promoted["preview"] == "v2"  # type: ignore[index]
    assert run_json(capsys, client, "projects", "auto-promote", name, "on")["production"] == "v2"  # type: ignore[index]
    run(capsys, client, "projects", "delete", name)
    assert name not in [p["name"] for p in run_json(capsys, client, "projects", "list")]  # type: ignore[union-attr]


def test_custom_models_datasets_runs(capsys: pytest.CaptureFixture[str], client: EndorClient, tmp_path: Path) -> None:
    f = tmp_path / "rows.jsonl"
    endor.data.save_rows(f, rows(4))
    ds = run_json(capsys, client, "datasets", "upload", "train", str(f))
    assert ds["n_rows"] == 4  # type: ignore[index]
    assert [d["name"] for d in run_json(capsys, client, "datasets", "list")] == ["train"]  # type: ignore[union-attr]
    r = client.custom_models.create("v1", base_model="jev-9b", run_name="cli-run")
    r.save_checkpoint().result()
    assert "cli-run" in run(capsys, client, "runs", "list")
    assert run_json(capsys, client, "runs", "show", r.id)["name"] == "cli-run"  # type: ignore[index]
    info = run_json(capsys, client, "custom-models", "info", "v1")
    assert info["step"] == 0 and info["live"] is False  # type: ignore[index]
    assert run_json(capsys, client, "custom-models", "set-ttl", "v1", "7200")["expires_at"]  # type: ignore[index]
    assert run_json(capsys, client, "custom-models", "set-ttl", "v1", "none")["expires_at"] is None  # type: ignore[index]
    assert run_json(capsys, client, "custom-models", "promote", "v1")["live"] is True  # type: ignore[index]
    assert [m["name"] for m in run_json(capsys, client, "custom-models", "list", "--live")] == ["v1"]  # type: ignore[union-attr]
    assert run_json(capsys, client, "custom-models", "list", "--inactive") == []
    assert run_json(capsys, client, "custom-models", "demote", "v1")["live"] is False  # type: ignore[index]
    ev = run_json(capsys, client, "eval", "v1", "train")
    assert ev["status"] == "completed" and ev["model"] == "v1"  # type: ignore[index]
    assert run_json(capsys, client, "eval", "jev-9b", "train")["model"] == "jev-9b"  # type: ignore[index]
    assert run_json(capsys, client, "runs", "close", r.id)["status"] in ("closing", "closed")  # type: ignore[index]
    run(capsys, client, "custom-models", "delete", "v1")
    assert run_json(capsys, client, "custom-models", "list") == []
    run(capsys, client, "datasets", "delete", "train")


def test_usage(capsys: pytest.CaptureFixture[str], client: EndorClient) -> None:
    out = run(capsys, client, "usage", "--start", "2026-10-01", "--end", "2026-10-03", "--csv")
    lines = out.splitlines()
    assert lines[0] == (
        "hour,kind,project,base_model,model,training_run_id,input_tokens,gpu_seconds,continuous_learning,"
        "price_per_mtok,base_cost_usd,continuous_learning_cost_usd,cost_usd"
    )
    assert lines[1].split(",")[8:12] == ["True", "0.3", "2.4e-05", "1.2e-05"]
    rows_json = run_json(capsys, client, "usage", "--start", "2026-10-01", "--end", "2026-10-03")
    assert isinstance(rows_json, list) and len(rows_json) == len(lines) - 1


def test_errors_exit_nonzero(capsys: pytest.CaptureFixture[str], client: EndorClient) -> None:
    assert cli.main(["projects", "create", "Bad Name", "--base-model", "jev-9b"], client=client) == 1
    assert "error: POST /v1/projects: 422 invalid_input" in capsys.readouterr().err
    assert cli.main(["custom-models", "info", "missing"], client=client) == 1
    assert "404" in capsys.readouterr().err


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and endor.__version__ in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main([])


def test_download(capsys: pytest.CaptureFixture[str], client: EndorClient, fake: FakeEndor, tmp_path: Path) -> None:
    with client.custom_models.create("v1", base_model="jev-9b") as r:
        r.save_checkpoint(include_optimizer=True).result()
    target = tmp_path / "m"
    out = run_json(capsys, client, "custom-models", "download", "v1", "-o", str(target))
    assert isinstance(out, list) and [f["name"] for f in out] == [
        "adapter_model.safetensors",
        "adapter_config.json",
        "endor_manifest.json",
    ]
    assert all(Path(f["path"]).parent == target and "url" not in f for f in out)
    run(capsys, client, "custom-models", "download", "v1", "-o", str(target), "--include-optimizer")
    assert (target / "optimizer.pt").is_file()
