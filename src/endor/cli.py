"""``endor``: inspect and manage what the SDK created. Training itself happens in Python.

    endor whoami
    endor base-models
    endor projects list | create NAME --base-model BASE | delete NAME
    endor projects versions NAME | promote NAME VERSION | auto-promote NAME on|off
    endor custom-models list [--production | --archived] | info NAME | promote NAME [--replace OTHER] | archive NAME
    endor custom-models download NAME [-o DIR] | set-ttl NAME SECONDS|none | delete NAME
    endor datasets list | upload NAME FILE | delete NAME
    endor runs list | show RUN_ID | close RUN_ID
    endor eval MODEL DATASET
    endor usage --start 2026-10-01 --end 2026-10-06 [--project P] [--csv]
    endor user-report "what you did, what happened, what you expected" [--kind bug|confusing|works] [--request-id REQ]

Projects are continuous learning projects; custom models are the ones you train with the SDK. Every command takes
-f json (or --format json), before or after the command. Credentials: ENDOR_API_KEY (and ENDOR_BASE_URL).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from . import _user_report
from . import data as D
from ._headers import sdk_context
from ._version import __version__
from .errors import APIError, EndorError, report_hint

if TYPE_CHECKING:
    from .client import EndorClient
    from .types import BaseModelInfo

__all__ = ["main", "build_parser"]


def _dump(obj: Any) -> Any:
    if hasattr(obj, "info_"):
        obj = obj.info_
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    return obj


def _out(obj: Any, fmt: str, empty: str = "") -> None:
    """Print ``obj``: indented JSON, or in table mode one ``key=value`` line per item (``empty`` for an empty list)."""
    obj = _dump(obj)
    if fmt == "json":
        print(json.dumps(obj, indent=1, default=str))
        return
    rows = obj if isinstance(obj, list) else [obj]
    if not rows and empty:
        print(empty)
    for r in rows:
        if isinstance(r, dict):
            print("  ".join(f"{k}={v}" for k, v in r.items() if not isinstance(v, (dict, list))))
        else:
            print(r)


def _block(title: str, values: dict[str, Any]) -> None:
    """Table mode: ``title:`` and one indented ``key=value`` line per entry (nothing when empty)."""
    if values:
        print(f"{title}:")
        for k, v in values.items():
            print(f"  {k}={v}")


def _accuracy(results: dict[str, Any] | None) -> None:
    """Table mode: an evaluation's accuracy, overall and per question, with the questions scored."""
    overall = (results or {}).get("overall")
    if not overall:
        return
    rows = [("overall", overall), *((results or {}).get("by_question") or {}).items()]
    width = max(len(name) for name, _ in rows)
    print(f"{'':<{width}}  {'n':>6}  accuracy")
    for name, m in rows:
        acc = m.get("accuracy")
        print(f"{name:<{width}}  {m.get('n', 0):>6}  {'-' if acc is None else f'{acc:.1%}':>8}")


def _base_models(bases: list[BaseModelInfo]) -> None:
    """Table mode: one line per base model with its Decision Index rank (``=`` marks a tie), measured speed, context,
    price per 1M input tokens and what it is meant for. ``-f json`` has every field."""
    if not bases:
        print("No base models")
        return
    rows = [("id", "index rank", "tokens/s", "context", "$/1M tokens", "meant for"), *map(_base_model_row, bases)]
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    for r in rows:
        cells = [r[0].ljust(widths[0]), *(r[i].rjust(widths[i]) for i in range(1, 5)), r[5]]
        print("  ".join(cells).rstrip())


def _base_model_row(b: BaseModelInfo) -> tuple[str, str, str, str, str, str]:
    i = b.decision_index
    if i is not None and i.rank is not None:
        rank = ("=" if i.tied else "") + str(i.rank) + (f"/{i.entrants}" if i.entrants else "")
    else:
        rank = "-" if b.leaderboard_rank is None else str(b.leaderboard_rank)
    return (
        b.id,
        rank,
        "-" if b.tokens_per_s is None else f"{b.tokens_per_s:,}",
        "-" if b.max_question_tokens is None else f"{b.max_question_tokens:,}",
        "-" if b.price_per_mtok_decide is None else f"${b.price_per_mtok_decide:g}",
        ", ".join(i.meant_for) if i is not None and i.meant_for else "-",
    )


FORMATS = ["table", "json"]


class _Subcommands(argparse._SubParsersAction):  # type: ignore[type-arg]
    """Subcommands that take ``-f/--format`` too, so it goes before or after any subcommand (``endor -f json whoami``,
    ``endor whoami -f json``). Given after, it overrides the one before."""

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        parser: argparse.ArgumentParser = super().add_parser(name, **kwargs)
        parser.register("action", "parsers", _Subcommands)
        _format_option(parser, argparse.SUPPRESS)  # not given here: the value before the subcommand stays
        return parser


def _format_option(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("-f", "--format", choices=FORMATS, default=default, help="table (default) or json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="endor", description="Endor CLI (training happens through the Python SDK)")
    p.register("action", "parsers", _Subcommands)
    _format_option(p, "table")
    p.add_argument("--version", action="version", version=f"endor {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    sub.add_parser("base-models")
    pr = sub.add_parser("projects").add_subparsers(dest="action", required=True)
    pr.add_parser("list")
    pc = pr.add_parser("create")
    pc.add_argument("name")
    pc.add_argument("--base-model", required=True, metavar="BASE", help="the base model it learns on, e.g. decider-2b")
    pr.add_parser("delete").add_argument("name")
    pr.add_parser("versions").add_argument("name")
    pp = pr.add_parser("promote")
    pp.add_argument("name")
    pp.add_argument("version", help="v1, v2, ...")
    pa = pr.add_parser("auto-promote")
    pa.add_argument("name")
    pa.add_argument("state", choices=["on", "off"])
    cm = sub.add_parser("custom-models").add_subparsers(dest="action", required=True)
    cl = cm.add_parser("list")
    state = cl.add_mutually_exclusive_group()
    state.add_argument("--production", action="store_const", const=True, dest="production")
    state.add_argument("--archived", action="store_const", const=False, dest="production")
    cm.add_parser("info").add_argument("name")
    cp = cm.add_parser("promote")
    cp.add_argument("name")
    cp.add_argument(
        "--replace", metavar="OTHER", help="a model in production to archive in the same step, when every slot is taken"
    )
    cm.add_parser("archive").add_argument("name")
    dl = cm.add_parser("download")
    dl.add_argument("name")
    dl.add_argument("-o", "--output", help="folder to write the files into (default: NAME)")
    dl.add_argument("--include-optimizer", action="store_true", help="also optimizer.pt, to resume training")
    ttl = cm.add_parser("set-ttl")
    ttl.add_argument("name")
    ttl.add_argument("seconds", help="seconds, or 'none' to keep the model")
    cm.add_parser("delete").add_argument("name")
    ds = sub.add_parser("datasets").add_subparsers(dest="action", required=True)
    ds.add_parser("list")
    up = ds.add_parser("upload")
    up.add_argument("name")
    up.add_argument("file", help=".jsonl or .json rows (see 'Data format' in the README)")
    ds.add_parser("delete").add_argument("name")
    runs = sub.add_parser("runs").add_subparsers(dest="action", required=True)
    runs.add_parser("list")
    runs.add_parser("show").add_argument("run_id")
    runs.add_parser("close").add_argument("run_id")
    ev = sub.add_parser("eval")
    ev.add_argument("model", help="a custom model's name or a base model id")
    ev.add_argument("dataset")
    us = sub.add_parser("usage")
    us.add_argument("--start", required=True, help="ISO date or datetime")
    us.add_argument("--end", required=True, help="ISO date or datetime")
    us.add_argument("--project")
    us.add_argument("--csv", action="store_true")
    rp = sub.add_parser("user-report", help="tell the Endor team what didn't work as expected, was unclear, or works")
    rp.add_argument("message", help="what you did, what happened and what you expected")
    rp.add_argument("--kind", choices=list(_user_report.KINDS), default="bug")
    rp.add_argument("--request-id", help="the req_... id of the request it is about")
    return p


def main(argv: Sequence[str] | None = None, client: EndorClient | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        with sdk_context(interface="cli"):
            if client is None:
                from .client import EndorClient

                client = EndorClient()
            if a.cmd == "user-report":
                _out(client.user_report(a.message, kind=a.kind, request_id=a.request_id), a.format)
                return 0
            _run(a, client)
    except EndorError as e:
        rid = e.request_id if isinstance(e, APIError) else None
        hint = "" if a.cmd == "user-report" else "\n" + report_hint(rid, cli=True)
        print(f"error: {e._text()}{hint}", file=sys.stderr)
        return 1
    return 0


def _run(a: argparse.Namespace, client: EndorClient) -> None:
    f = a.format
    if a.cmd == "whoami":
        me = {**_dump(client.whoami()), "balance_usd": client.balance()}
        _out(me, f)
        if f == "table":
            _block("limits", me["limits"])
    elif a.cmd == "base-models":
        bases = client.base_models()
        if f == "json":
            _out(bases, f)
        else:
            _base_models(bases)
    elif a.cmd == "projects":
        if a.action == "list":
            _out(client.projects.list(), f, "No projects")
        elif a.action == "create":
            _out(client.projects.create(a.name, base_model=a.base_model), f)
        elif a.action == "versions":
            _out(client.projects.get(a.name).versions(), f, "No versions")
        elif a.action == "promote":
            _out(client.projects.get(a.name).promote(a.version), f)
        elif a.action == "auto-promote":
            _out(client.projects.get(a.name).set_auto_promote(a.state == "on"), f)
        else:
            client.projects.get(a.name).delete()
    elif a.cmd == "custom-models":
        cm = client.custom_models
        if a.action == "list":
            _out(cm.list(production=a.production), f, "No custom models")
        elif a.action == "info":
            _out(cm.get(a.name), f)
        elif a.action == "promote":
            _out(cm.promote(a.name, replace=a.replace), f)
        elif a.action == "archive":
            _out(cm.archive(a.name), f)
        elif a.action == "set-ttl":
            _out(cm.set_ttl(a.name, None if a.seconds == "none" else int(a.seconds)), f)
        elif a.action == "delete":
            cm.delete(a.name)
        elif a.action == "download":
            _out(cm.download(a.name, a.output or a.name, include_optimizer=a.include_optimizer), f)
    elif a.cmd == "datasets":
        if a.action == "list":
            _out(client.datasets.list(), f, "No datasets")
        elif a.action == "upload":
            _out(client.datasets.upload(a.name, D.load_rows(a.file)), f)
        else:
            client.datasets.delete(a.name)
    elif a.cmd == "runs":
        if a.action == "list":
            _out(client.runs.list(), f, "No runs")
        else:
            run = client.runs.get(a.run_id)
            _out(run.close() if a.action == "close" else run, f)
    elif a.cmd == "eval":
        ev = client.evaluations.create(a.model, a.dataset).result()
        _out(ev, f)
        if f == "table":
            _accuracy(ev.results)
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
            _out(rows, f, "No usage")


if __name__ == "__main__":
    raise SystemExit(main())
