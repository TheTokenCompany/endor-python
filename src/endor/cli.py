"""``endor``: inspect and manage what the SDK created. Training itself happens in Python.

    endor whoami
    endor base-models
    endor projects list | create NAME --base-model BASE [--kind custom|managed] | delete NAME
    endor datasets list PROJECT | upload PROJECT NAME FILE | delete PROJECT NAME
    endor runs list PROJECT | show RUN_ID | close RUN_ID
    endor models list PROJECT | info MODEL | download MODEL [-o DIR] | set-ttl MODEL SECONDS|none | delete MODEL
    endor eval PROJECT MODEL DATASET
    endor usage --start 2026-10-01 --end 2026-10-06 [--project P] [--csv]

MODEL is "<project>/<name>" ("<project>/base" for the project's base model). Every command takes -f json.
Credentials: ENDOR_API_KEY (and ENDOR_BASE_URL).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from . import data as D
from ._headers import sdk_context
from ._version import __version__
from .errors import EndorError

if TYPE_CHECKING:
    from .client import EndorClient

__all__ = ["main", "build_parser"]


def _dump(obj: Any) -> Any:
    if hasattr(obj, "info_"):
        obj = obj.info_
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    return obj


def _out(obj: Any, fmt: str) -> None:
    obj = _dump(obj)
    if fmt == "json":
        print(json.dumps(obj, indent=1, default=str))
        return
    rows = obj if isinstance(obj, list) else [obj]
    for r in rows:
        if isinstance(r, dict):
            print("  ".join(f"{k}={v}" for k, v in r.items() if not isinstance(v, (dict, list))))
        else:
            print(r)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="endor", description="Endor CLI (training happens through the Python SDK)")
    p.add_argument("-f", "--format", choices=["table", "json"], default="table")
    p.add_argument("--version", action="version", version=f"endor {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    sub.add_parser("base-models")
    pr = sub.add_parser("projects").add_subparsers(dest="action", required=True)
    pr.add_parser("list")
    pc = pr.add_parser("create")
    pc.add_argument("name")
    pc.add_argument("--base-model", required=True, help="the project's base model, e.g. decider-2b (never changes)")
    pc.add_argument(
        "--kind",
        choices=["custom", "managed"],
        default="custom",
        help="custom: you train models with the SDK; managed: Endor trains from the project's decisions (+50%%)",
    )
    pr.add_parser("delete").add_argument("name")
    ds = sub.add_parser("datasets").add_subparsers(dest="action", required=True)
    ds.add_parser("list").add_argument("project")
    up = ds.add_parser("upload")
    up.add_argument("project")
    up.add_argument("name")
    up.add_argument("file", help=".jsonl or .json rows (see 'Data format' in the README)")
    dd = ds.add_parser("delete")
    dd.add_argument("project")
    dd.add_argument("name")
    runs = sub.add_parser("runs").add_subparsers(dest="action", required=True)
    runs.add_parser("list").add_argument("project")
    runs.add_parser("show").add_argument("run_id")
    runs.add_parser("close").add_argument("run_id")
    md = sub.add_parser("models").add_subparsers(dest="action", required=True)
    md.add_parser("list").add_argument("project")
    md.add_parser("info").add_argument("model")
    dl = md.add_parser("download")
    dl.add_argument("model")
    dl.add_argument("-o", "--output", help="folder to write the files into (default: <project>-<name>)")
    dl.add_argument("--include-optimizer", action="store_true", help="also optimizer.pt, to resume training")
    ttl = md.add_parser("set-ttl")
    ttl.add_argument("model")
    ttl.add_argument("seconds", help="seconds, or 'none' to keep the model")
    md.add_parser("delete").add_argument("model")
    ev = sub.add_parser("eval")
    ev.add_argument("project")
    ev.add_argument("model")
    ev.add_argument("dataset")
    us = sub.add_parser("usage")
    us.add_argument("--start", required=True, help="ISO date or datetime")
    us.add_argument("--end", required=True, help="ISO date or datetime")
    us.add_argument("--project")
    us.add_argument("--csv", action="store_true")
    return p


def main(argv: Sequence[str] | None = None, client: EndorClient | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        with sdk_context(interface="cli"):
            if client is None:
                from .client import EndorClient

                client = EndorClient()
            _run(a, client)
    except EndorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _run(a: argparse.Namespace, client: EndorClient) -> None:
    f = a.format

    def project_of(model: str) -> Any:
        if "/" not in model:
            raise EndorError("MODEL must be '<project>/<name>'")
        return client.projects.get(model.split("/", 1)[0])

    if a.cmd == "whoami":
        _out(client.whoami(), f)
    elif a.cmd == "base-models":
        _out(client.base_models(), f)
    elif a.cmd == "projects":
        if a.action == "list":
            _out(client.projects.list(), f)
        elif a.action == "create":
            _out(client.projects.create(a.name, base_model=a.base_model, kind=a.kind), f)
        else:
            client.projects.get(a.name).delete()
    elif a.cmd == "datasets":
        p = client.projects.get(a.project)
        if a.action == "list":
            _out(p.datasets.list(), f)
        elif a.action == "upload":
            _out(p.datasets.upload(a.name, D.load_rows(a.file)), f)
        else:
            p.datasets.delete(a.name)
    elif a.cmd == "runs":
        if a.action == "list":
            _out(client.projects.get(a.project).runs.list(), f)
        else:
            from .runs import Run

            run = Run(client._t, client._t.request("GET", f"/v1/runs/{a.run_id}", method_name="project.runs.get"))
            _out(run.close() if a.action == "close" else run, f)
    elif a.cmd == "models":
        if a.action == "list":
            _out(client.projects.get(a.project).models.list(), f)
        else:
            models = project_of(a.model).models
            if a.action == "info":
                _out(models.get(a.model), f)
            elif a.action == "set-ttl":
                _out(models.set_ttl(a.model, None if a.seconds == "none" else int(a.seconds)), f)
            elif a.action == "delete":
                models.delete(a.model)
            elif a.action == "download":
                target = a.output or a.model.replace("/", "-")
                _out(models.download(a.model, target, include_optimizer=a.include_optimizer), f)
    elif a.cmd == "eval":
        _out(client.projects.get(a.project).evaluate(a.model, a.dataset).result(), f)
    elif a.cmd == "usage":
        rows = client.usage(datetime.fromisoformat(a.start), datetime.fromisoformat(a.end), a.project)
        if a.csv:
            w = csv.writer(sys.stdout)
            cols = [
                "hour",
                "kind",
                "project",
                "base_model",
                "model",
                "training_run_id",
                "input_tokens",
                "gpu_seconds",
                "continuous_learning",
                "price_per_mtok",
                "base_cost_usd",
                "continuous_learning_cost_usd",
                "cost_usd",
            ]
            w.writerow(cols)
            for u in rows:
                w.writerow(["" if v is None else v for v in (u.model_dump(mode="json")[c] for c in cols)])
        else:
            _out(rows, f)


if __name__ == "__main__":
    raise SystemExit(main())
