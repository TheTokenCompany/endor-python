from __future__ import annotations

import argparse
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
    assert "org_id=" in out and "balance_usd=" in out and "continuous_learning=" in out
    me = run_json(capsys, client, "whoami")
    assert isinstance(me, dict) and me["org_id"] and "limits" in me and "balance_usd" in me
    assert isinstance(me["continuous_learning"], bool)
    lines = out.splitlines()
    assert lines[1] == "limits:" and f"  max_active_runs={me['limits']['max_active_runs']}" in lines[2:]
    assert len(lines) == 2 + len(me["limits"])  # every limit, one per line
    bases = run_json(capsys, client, "base-models")
    assert isinstance(bases, list) and {b["id"] for b in bases} >= {"autotrust/jev-9b"}
    assert all("decision_index" in b and "tokens_per_s" in b and "trainer_gpu" not in b for b in bases)
    table = run(capsys, client, "base-models").splitlines()
    assert table[0].split() == ["id", "rank", "tokens/s", "context", "$/1M", "tokens", "meant", "for"]
    assert [line.split()[0] for line in table[1:]] == [b["id"] for b in bases]  # one line per base, in order


def test_projects(capsys: pytest.CaptureFixture[str], client: EndorClient, fake: FakeEndor) -> None:
    name = unique("cli")
    created = run_json(capsys, client, "projects", "create", name, "--base-model", "autotrust/jev-9b")
    assert created["name"] == name and created["base_model"] == "autotrust/jev-9b"  # type: ignore[index]
    assert created["auto_promote"] is True  # type: ignore[index]
    assert name in [p["name"] for p in run_json(capsys, client, "projects", "list")]  # type: ignore[union-attr]
    assert run_json(capsys, client, "projects", "auto-promote", name, "off")["auto_promote"] is False  # type: ignore[index]
    fake.add_version(name)
    fake.add_version(name)  # auto-promote is off: v2 is the preview, nothing live yet
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
    r = client.custom_models.create("v1", base_model="autotrust/jev-9b", run_name="cli-run")
    r.save_checkpoint().result()
    assert "cli-run" in run(capsys, client, "runs", "list")
    assert run_json(capsys, client, "runs", "show", r.id)["name"] == "cli-run"  # type: ignore[index]
    info = run_json(capsys, client, "custom-models", "info", "v1")
    assert info["step"] == 0 and info["state"] == "archived"  # type: ignore[index]
    assert run_json(capsys, client, "custom-models", "set-ttl", "v1", "7200")["expires_at"]  # type: ignore[index]
    assert run_json(capsys, client, "custom-models", "set-ttl", "v1", "none")["expires_at"] is None  # type: ignore[index]
    assert run_json(capsys, client, "custom-models", "promote", "v1")["production"] is True  # type: ignore[index]
    assert [m["name"] for m in run_json(capsys, client, "custom-models", "list", "--production")] == ["v1"]  # type: ignore[union-attr]
    assert run_json(capsys, client, "custom-models", "list", "--archived") == []
    assert run_json(capsys, client, "custom-models", "archive", "v1")["production"] is False  # type: ignore[index]
    ev = run_json(capsys, client, "eval", "v1", "train")
    assert ev["status"] == "completed" and ev["model"] == "v1"  # type: ignore[index]
    assert run_json(capsys, client, "eval", "autotrust/jev-9b", "train")["model"] == "autotrust/jev-9b"  # type: ignore[index]
    assert run_json(capsys, client, "runs", "close", r.id)["status"] in ("closing", "closed")  # type: ignore[index]
    run(capsys, client, "custom-models", "delete", "v1")
    assert run_json(capsys, client, "custom-models", "list") == []
    run(capsys, client, "datasets", "delete", "train")


def test_tables_show_serving_and_accuracy(
    capsys: pytest.CaptureFixture[str], client: EndorClient, fake: FakeEndor
) -> None:
    client.datasets.upload("heldout", rows(4))
    with client.custom_models.create("v1", base_model="autotrust/jev-9b") as r:
        r.save_checkpoint().result()
    assert "serving=None" in run(capsys, client, "custom-models", "info", "v1")
    client.custom_models.promote("v1")
    assert "  serving=live  serving_error=None  " in run(capsys, client, "custom-models", "list")
    fake.models["v1"]["_serving"], fake.models["v1"]["_serving_error"] = "failed", "no GPU took it"
    assert "serving=failed  serving_error=no GPU took it" in run(capsys, client, "custom-models", "info", "v1")
    assert run_json(capsys, client, "custom-models", "info", "v1")["serving"] == "failed"  # type: ignore[index]
    lines = run(capsys, client, "eval", "v1", "heldout").splitlines()
    assert "status=completed" in lines[0]
    assert [x.split() for x in lines[1:]] == [
        ["n", "accuracy"],
        ["overall", "8", "50.0%"],
        ["dept", "4", "50.0%"],
        ["urgent", "4", "50.0%"],
    ]
    client.custom_models.archive("v1")


def test_empty_lists_say_so(capsys: pytest.CaptureFixture[str], client: EndorClient, fake: FakeEndor) -> None:
    assert run(capsys, client, "custom-models", "list") == "No custom models\n"
    assert run(capsys, client, "custom-models", "list", "--production") == "No custom models\n"
    assert run(capsys, client, "datasets", "list") == "No datasets\n"
    assert run(capsys, client, "runs", "list") == "No runs\n"
    assert run(capsys, client, "projects", "list") == "No projects\n"
    assert run_json(capsys, client, "custom-models", "list") == []  # JSON stays a list


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
    assert cli.main(["projects", "create", "Bad Name", "--base-model", "autotrust/jev-9b"], client=client) == 1
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
    with client.custom_models.create("v1", base_model="autotrust/jev-9b") as r:
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


# Every command once, with the arguments it needs.
COMMANDS = [
    ["whoami"],
    ["base-models"],
    ["projects", "list"],
    ["projects", "create", "p", "--base-model", "autotrust/jev-9b"],
    ["projects", "delete", "p"],
    ["projects", "versions", "p"],
    ["projects", "promote", "p", "v1"],
    ["projects", "auto-promote", "p", "on"],
    ["custom-models", "list", "--production"],
    ["custom-models", "info", "m"],
    ["custom-models", "promote", "m", "--replace", "n"],
    ["custom-models", "archive", "m"],
    ["custom-models", "download", "m", "-o", "out", "--include-optimizer"],
    ["custom-models", "set-ttl", "m", "none"],
    ["custom-models", "delete", "m"],
    ["datasets", "list"],
    ["datasets", "upload", "d", "rows.jsonl"],
    ["datasets", "delete", "d"],
    ["runs", "list"],
    ["runs", "show", "run_1"],
    ["runs", "close", "run_1"],
    ["eval", "m", "d"],
    ["usage", "--start", "2026-10-01", "--end", "2026-10-02", "--csv"],
    ["user-report", "it works", "--kind", "works"],
]


def _commands(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Every command the parser has, as its words (``("projects", "list")``)."""
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not subs:
        return {prefix}
    return {c for name, sub in subs[0].choices.items() for c in _commands(sub, (*prefix, name))}


def test_the_list_has_every_command() -> None:
    p = cli.build_parser()
    listed = {tuple(w for w in (a.cmd, getattr(a, "action", None)) if w) for a in map(p.parse_args, COMMANDS)}
    assert listed == _commands(p)


@pytest.mark.parametrize("argv", COMMANDS, ids=" ".join)
def test_every_command_takes_the_format_before_or_after_it(argv: list[str]) -> None:
    p = cli.build_parser()
    assert p.parse_args(argv).format == "table"
    assert p.parse_args(["-f", "json", *argv]).format == "json"
    assert p.parse_args([*argv, "-f", "json"]).format == "json"
    assert p.parse_args([*argv, "--format", "json"]).format == "json"
    assert p.parse_args(["-f", "json", *argv, "-f", "table"]).format == "table"  # the one after wins
    with pytest.raises(SystemExit):
        p.parse_args([*argv, "-f", "yaml"])


def test_format_after_the_command(capsys: pytest.CaptureFixture[str], client: EndorClient) -> None:
    after = json.loads(run(capsys, client, "whoami", "-f", "json"))
    assert after == json.loads(run(capsys, client, "-f", "json", "whoami")) and "limits" in after
    assert json.loads(run(capsys, client, "custom-models", "list", "--production", "--format", "json")) == []
    assert json.loads(run(capsys, client, "projects", "-f", "json", "list")) == []
    assert run(capsys, client, "-f", "json", "projects", "list", "-f", "table") == "No projects\n"
