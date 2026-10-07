"""An in-memory fake of the Endor API, served through ``httpx.MockTransport``.

It implements the public contract the SDK relies on: bearer auth, the error envelope, strict sequence numbers with
idempotent retries, ``Idempotency-Key`` on creates, long-polled futures (optionally pending for N polls), the
decision answer formulas, model ids through a project (``<project>``, ``<project>/base``, ``<project>/<name>``),
the count limits (409 ``limit_reached``), ``no_gradients``, 402 for a blocked org, and fault injection for retry
tests. It is a test double, not a reference server.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs

import httpx

FILES_HOST = "files.endor.test"
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")

LIMITS: dict[str, float] = {
    "max_projects_per_org": 7,
    "max_api_keys_per_org": 20,
    "max_active_runs": 4,
    "max_models_per_project": 50,
    "max_dataset_rows": 50_000,
    "max_pending_ops": 64,
    "decisions_per_second": 50.0,
    "decisions_burst": 100,
    "downloads_per_minute": 30,
    "dataset_uploads_per_minute": 20,
    "runs_per_minute": 20,
    "evaluations_per_minute": 30,
}
MAX_ACTIVE_RUNS = int(LIMITS["max_active_runs"])
BASE = "base"  # "<project>/base": the project's base model
BASE_MODELS: dict[str, dict[str, Any]] = {
    "pplx-decider-v1.1-27b": {
        "max_options": 255,
        "trainable": True,
        "default_rank": 16,
        "lora_targets": ["attn", "mlp"],
    },
    "jev-9b": {
        "max_options": 16,
        "trainable": True,
        "default_rank": 16,
        "lora_targets": ["attn", "mlp"],
    },
    "decider-2b": {
        "max_options": 255,
        "trainable": True,
        "default_rank": 16,
        "lora_targets": ["attn", "mlp"],
    },
    "gev-26b": {
        "max_options": 16,
        "trainable": True,
        "default_rank": 16,
        "lora_targets": ["attn", "mlp"],
    },
    "gliner2.5-decide": {
        "max_options": 64,
        "trainable": True,
        "default_rank": 16,
        "lora_targets": ["attn", "mlp"],
    },
}


class HTTPError(Exception):
    def __init__(self, status: int, code: str, message: str, param: str | None = None, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.param, self.headers = status, code, message, param, headers or {}


def now() -> datetime:
    return datetime.now(timezone.utc)


def option_keys(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def check_question(q: Any, where: str) -> None:
    if not isinstance(q, dict) or q.get("type") not in ("noul", "choice", "score"):
        raise HTTPError(422, "invalid_input", "bad question type", where)
    if q["type"] == "choice" and not (isinstance(q.get("criteria"), dict) and 1 <= len(q["criteria"]) <= 255):
        raise HTTPError(422, "invalid_input", "a choice has 1..255 options", where)
    if q["type"] == "score" and not (isinstance(q.get("criteria"), list) and 2 <= len(q["criteria"]) <= 10):
        raise HTTPError(422, "invalid_input", "a score has 2..10 levels", where)


def label_target(q: dict, label: Any) -> dict:
    keys = option_keys(q)
    if isinstance(label, dict):
        probs = {str(k): float(v) for k, v in label.items()}
    elif isinstance(label, list):
        probs = dict(zip(keys, map(float, label), strict=False))
    elif q["type"] == "noul" and isinstance(label, bool):
        return {"label": "true" if label else "false"}
    elif q["type"] == "noul" and isinstance(label, (int, float)) and 0 <= label <= 1:
        probs = {"false": 1 - float(label), "true": float(label)}
    elif q["type"] == "score" and (
        (isinstance(label, int) and not isinstance(label, bool)) or (isinstance(label, str) and label.isdigit())
    ):
        if str(label) not in keys:
            raise ValueError(f"{str(label)!r} is not an option of this question")
        return {"label": str(label)}
    elif q["type"] == "choice" and isinstance(label, str):
        if label not in keys:
            raise ValueError(f"{label!r} is not an option")
        return {"label": label}
    else:
        raise ValueError(f"label {label!r} does not fit a {q['type']} question")
    if set(probs) != set(keys) or min(probs.values()) < 0 or not math.isclose(sum(probs.values()), 1, abs_tol=0.02):
        raise ValueError("probs must cover the options and sum to 1")
    return {"probs": probs}


def check_datum(d: dict, base: dict, where: str) -> None:
    check_question(d.get("question"), where)
    t = d.get("target") or {}
    keys = set(option_keys(d["question"]))
    if ("label" in t) == ("probs" in t):
        raise HTTPError(422, "invalid_datum", f"{where}: target needs exactly one of label or probs")
    if "label" in t and t["label"] not in keys:
        raise HTTPError(422, "invalid_datum", f"{where}: target.label is not an option")
    if "probs" in t and set(t["probs"]) != keys:
        raise HTTPError(422, "invalid_datum", f"{where}: target.probs must cover exactly the options")
    if d.get("weight", 1.0) < 0:
        raise HTTPError(422, "invalid_datum", f"{where}: weight must be >= 0")
    if len(keys) > base["max_options"]:
        raise HTTPError(422, "invalid_datum", f"{where}: more options than the base reads")


def uniform(q: dict) -> list[float]:
    n = len(option_keys(q))
    return [1 / n] * n


def answer_from_probs(q: dict, probs: list[float]) -> dict:
    keys = option_keys(q)
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs[1]}
    n = len(keys)
    m = max(range(n), key=lambda i: probs[i])
    if q["type"] == "choice":
        conf = 1.0 if n == 1 else (probs[m] - 1 / n) / (1 - 1 / n)
        probabilities = dict(zip(keys, probs, strict=False))
        return {"type": "choice", "choice": keys[m], "probabilities": probabilities, "confidence": conf}
    mad = sum(p * abs(i - m) for i, p in enumerate(probs))
    mad_unif = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs)),
        "legend": dict(zip(keys, q["criteria"], strict=False)),
        "probabilities": dict(zip(keys, probs, strict=False)),
        "confidence": max(0.0, 1 - mad / mad_unif),
    }


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: Any
    params: dict[str, str]


@dataclass
class FakeEndor:
    """State plus the request handler. Mutate the knobs from tests."""

    api_keys: set[str] = field(default_factory=lambda: {"edk_test"})
    future_polls: int = 0
    """How many polls a new future stays pending before completing (0 = completed at once)."""
    provisioning_polls: int = 0
    """How many polls a run's ready future stays pending."""
    fail_next: list[Any] = field(default_factory=list)
    """Queued faults applied to the next requests: an ``HTTPError`` or an ``httpx.TransportError``."""
    requests: list[Recorded] = field(default_factory=list)
    file_requests: list[Recorded] = field(default_factory=list)  # GETs of presigned file links
    file_links: dict[str, tuple[str, str]] = field(default_factory=dict)  # token -> (model id, file name)
    corrupt_files: bool = False
    blocked: bool = False
    """The org's balance is used up: paid calls answer 402 insufficient_balance."""

    projects: dict[str, dict] = field(default_factory=dict)
    datasets: dict[tuple[str, str], dict] = field(default_factory=dict)
    runs: dict[str, dict] = field(default_factory=dict)
    models: dict[str, dict] = field(default_factory=dict)
    futures: dict[str, dict] = field(default_factory=dict)
    evaluations: dict[str, dict] = field(default_factory=dict)
    metrics: dict[str, list[dict]] = field(default_factory=dict)
    ops: dict[tuple[str, int], tuple[str, str]] = field(default_factory=dict)  # (run, seq) -> (body hash, future)
    idempotency: dict[str, tuple[int, Any]] = field(default_factory=dict)
    _ids: Counter = field(default_factory=Counter)

    # ------------------------------------------------------------------ helpers
    def new_id(self, prefix: str) -> str:
        self._ids[prefix] += 1
        return f"{prefix}_{self._ids[prefix]:04d}"

    def future(
        self, kind: str, result: dict | None = None, *, error: dict | None = None, pending: int | None = None
    ) -> dict:
        f = {
            "id": self.new_id("fut"),
            "kind": kind,
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": now().isoformat(),
            "completed_at": None,
            "_result": result,
            "_error": error,
            "_polls": self.future_polls if pending is None else pending,
        }
        self.futures[f["id"]] = f
        return f

    def poll(self, f: dict) -> dict:
        if f["status"] in ("pending", "running"):
            if f["_polls"] <= 0:
                if f["_error"] is not None:
                    f["status"], f["error"] = "failed", f["_error"]
                else:
                    f["status"], f["result"] = "completed", f["_result"]
                    if f["kind"] == "provision_run" and f["_result"]["run_id"] in self.runs:
                        self.runs[f["_result"]["run_id"]]["status"] = "ready"
                f["completed_at"] = now().isoformat()
            else:
                f["_polls"] -= 1
                f["status"] = "running"
        return {k: v for k, v in f.items() if not k.startswith("_")}

    def fail_future(self, future_id: str, code: str, message: str = "boom") -> None:
        f = self.futures[future_id]
        f["_error"], f["_result"] = {"code": code, "message": message}, None

    @staticmethod
    def project_view(p: dict, self_: FakeEndor) -> dict:
        live = self_.live_model(p)
        return {
            "name": p["name"],
            "description": p["description"],
            "kind": p["kind"],
            "base_model": p["base_model"],
            "live_model": None if p["base_model"] is None else (live or f"{p['name']}/{BASE}"),
            "paused": p["paused"] if p["kind"] == "managed" else None,
            "created_at": p["created_at"],
            "n_datasets": sum(1 for (pr, _) in self_.datasets if pr == p["name"]),
            "n_runs": sum(1 for r in self_.runs.values() if r["project"] == p["name"]),
            "n_models": sum(1 for m in self_.models.values() if m["project"] == p["name"])
            + (p["base_model"] is not None),
        }

    def live_model(self, p: dict) -> str | None:
        """The saved model "<project>" serves, or None for its base."""
        mid = p["live_model"]
        return mid if mid in self.models and self.models[mid]["_ready"] else None

    def promote(self, project: str, name: str) -> None:
        """What set_live does: make a model (or the base) what "<project>" serves."""
        self.project(project)["live_model"] = None if name == BASE else f"{project}/{name}"

    def require_custom(self, project: str, what: str) -> None:
        """409 wrong_project_kind for a managed project (Endor trains it)."""
        if self.project(project)["kind"] == "managed":
            raise HTTPError(
                409, "wrong_project_kind", f"Project {project} is managed: Endor trains it, so {what} isn't available."
            )

    def base_model_view(self, p: dict) -> dict:
        return {
            "id": f"{p['name']}/{BASE}",
            "project": p["name"],
            "name": BASE,
            "live": self.live_model(p) is None,
            "training_run_id": None,
            "base_model": p["base_model"],
            "contract": "letters-16",
            "parent_model": None,
            "step": 0,
            "has_optimizer": False,
            "size_bytes": None,
            "expires_at": None,
            "user_metadata": {},
            "created_at": p["created_at"],
        }

    def project(self, name: str) -> dict:
        if name not in self.projects:
            raise HTTPError(404, "not_found", f"no project {name}")
        return self.projects[name]

    def run(self, run_id: str) -> dict:
        if run_id not in self.runs:
            raise HTTPError(404, "not_found", f"no run {run_id}")
        return self.runs[run_id]

    @staticmethod
    def run_view(r: dict) -> dict:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    def model_view(self, m: dict) -> dict:
        live = self.live_model(self.projects[m["project"]]) == m["id"] if m["project"] in self.projects else False
        return {**{k: v for k, v in m.items() if not k.startswith("_")}, "live": live}

    @staticmethod
    def file_bytes(mid: str, name: str) -> bytes:
        return f"{mid}:{name}:".encode() * 1000

    def serve_file(self, request: httpx.Request) -> httpx.Response:
        """A presigned storage GET: no Endor auth, the token in the path is the whole credential."""
        token, _, name = request.url.path.strip("/").partition("/")
        hit = self.file_links.get(token)
        if hit is None or hit[1] != name or request.method != "GET":
            return httpx.Response(403, text="<Error><Code>AccessDenied</Code></Error>")
        data = self.file_bytes(*hit)
        if self.corrupt_files:
            data = data[:-1] + b"!"
        return httpx.Response(200, content=data, headers={"content-type": "application/octet-stream"})

    # ------------------------------------------------------------------ handler
    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == FILES_HOST:
            self.file_requests.append(Recorded(request.method, request.url.path, dict(request.headers), None, {}))
            return self.serve_file(request)
        raw = request.content
        if raw and request.headers.get("content-encoding") == "gzip":
            raw = gzip.decompress(raw)
        body = json.loads(raw) if raw else None
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        self.requests.append(Recorded(request.method, request.url.path, dict(request.headers), body, params))
        if self.fail_next:
            fault = self.fail_next.pop(0)
            if isinstance(fault, HTTPError):
                return self._error(fault)
            if fault is not None:  # None: let this request through
                raise fault
        try:
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] not in self.api_keys:
                raise HTTPError(401, "unauthorized", "bad API key")
            idem = request.headers.get("idempotency-key")
            if idem and request.method == "POST" and idem in self.idempotency:
                status, payload = self.idempotency[idem]
                return self._json(status, payload, replay=True)
            status, payload = self.route(request.method, request.url.path, body, params)
            if idem and request.method == "POST" and 200 <= status < 300:
                self.idempotency[idem] = (status, payload)
            return self._json(status, payload)
        except HTTPError as e:
            return self._error(e)

    def _json(self, status: int, payload: Any, replay: bool = False) -> httpx.Response:
        headers = {"x-request-id": self.new_id("req")}
        if replay:
            headers["idempotent-replayed"] = "true"
        if status == 204:
            return httpx.Response(204, headers=headers)
        return httpx.Response(status, json=payload, headers=headers)

    def _error(self, e: HTTPError) -> httpx.Response:
        err = {"code": e.code, "message": e.message}
        if e.param:
            err["param"] = e.param
        return httpx.Response(e.status, json={"error": err}, headers={"x-request-id": self.new_id("req"), **e.headers})

    # ------------------------------------------------------------------ routing
    def route(self, method: str, path: str, body: Any, params: dict[str, str]) -> tuple[int, Any]:
        parts = [p for p in path.split("/") if p]
        if parts[:1] != ["v1"]:
            raise HTTPError(404, "not_found", "no such route")
        p = parts[1:]
        m = method
        if p == ["systemone"] and m == "POST":
            return 200, self.systemone(body)
        if p == ["models"] and m == "GET":
            return 200, self.list_models()
        if p == ["base_models"] and m == "GET":
            return 200, self.list_base_models()
        if p == ["whoami"] and m == "GET":
            return 200, {
                "user_id": None,
                "org_id": "org_test",
                "key_id": "key_test",
                "limits": dict(LIMITS),
            }
        if p == ["usage"] and m == "GET":
            start, end = datetime.fromisoformat(params["starting_on"]), datetime.fromisoformat(params["ending_before"])
            if end - start > timedelta(days=14):
                raise HTTPError(422, "invalid_input", "at most 14 days per call", "ending_before")
            hour = start.replace(minute=0, second=0, microsecond=0).isoformat()
            common = {"hour": hour, "project": params.get("project"), "base_model": "jev-9b"}
            return 200, [
                {
                    **common,
                    "kind": "decide",
                    "model": None,
                    "training_run_id": None,
                    "input_tokens": 120,
                    "gpu_seconds": None,
                    "continuous_learning": True,
                    "price_per_mtok": 0.3,
                    "base_cost_usd": 0.000024,
                    "continuous_learning_cost_usd": 0.000012,
                    "cost_usd": 0.000036,
                },
                {
                    **common,
                    "kind": "train",
                    "model": None,
                    "training_run_id": "run_0001",
                    "input_tokens": None,
                    "gpu_seconds": 360,
                    "continuous_learning": None,
                    "price_per_mtok": None,
                    "base_cost_usd": None,
                    "continuous_learning_cost_usd": None,
                    "cost_usd": 0.3,
                },
            ]
        # projects
        if p == ["projects"] and m == "POST":
            return 201, self.create_project(body)
        if p == ["projects"] and m == "GET":
            items = sorted(self.projects.values(), key=lambda x: x["created_at"], reverse=True)
            return 200, self.page([self.project_view(x, self) for x in items], params)
        if len(p) == 2 and p[0] == "projects":
            if m == "GET":
                return 200, self.project_view(self.project(p[1]), self)
            if m == "PATCH":
                proj = self.project(p[1])
                if "description" in body:
                    proj["description"] = body["description"]
                for field in ("kind", "base_model"):
                    if field in body:
                        raise HTTPError(422, "invalid_input", f"A project's {field} can't change.", field)
                if body.get("paused") is not None:
                    if proj["kind"] != "managed":
                        raise HTTPError(409, "wrong_project_kind", f"Project {p[1]} is custom: only managed pause.")
                    proj["paused"] = body["paused"]
                return 200, self.project_view(proj, self)
            if m == "DELETE":
                self.project(p[1])
                if any(r["project"] == p[1] and r["status"] in ("provisioning", "ready") for r in self.runs.values()):
                    raise HTTPError(409, "run_active", "close the project's active runs first")
                for k in [k for k in self.datasets if k[0] == p[1]]:
                    del self.datasets[k]
                for k in [k for k, v in self.models.items() if v["project"] == p[1]]:
                    del self.models[k]
                del self.projects[p[1]]
                return 204, None
        if len(p) == 3 and p[0] == "projects" and p[2] == "live" and m == "POST":
            proj = self.project(p[1])
            self.require_custom(p[1], "set_live")
            name = str(body.get("model", "")).removeprefix(f"{p[1]}/")
            if name == BASE:
                if proj["base_model"] is None:
                    raise HTTPError(409, "no_base_model", f"Project {p[1]} has no base model yet.", "model")
            elif not (f"{p[1]}/{name}" in self.models and self.models[f"{p[1]}/{name}"]["_ready"]):
                raise HTTPError(404, "not_found", f"No model {p[1]}/{name}.", "name")
            self.promote(p[1], name)
            return 200, self.project_view(proj, self)
        if len(p) >= 3 and p[0] == "projects":
            project, sub = p[1], p[2]
            self.project(project)
            if sub == "datasets":
                if m == "POST":
                    self.require_custom(project, "datasets")
                return self.datasets_route(m, project, p[3:], body, params)
            if sub == "runs":
                if m == "POST" and len(p) == 3:
                    self.require_custom(project, "training runs")
                    return 201, self.create_run(project, body)
                if m == "GET" and len(p) == 3:
                    items = sorted(
                        (
                            r
                            for r in self.runs.values()
                            if r["project"] == project and (params.get("tag") is None or params["tag"] in r["tags"])
                        ),
                        key=lambda r: r["created_at"],
                        reverse=True,
                    )
                    return 200, self.page([self.run_view(r) for r in items], params)
            if sub == "models":
                return self.models_route(m, project, p[3:], body, params)
            if sub == "evaluations":
                if m == "POST":
                    self.require_custom(project, "evaluations")
                    return 202, self.create_evaluation(project, body)
                if m == "GET":
                    items = sorted(
                        (
                            e
                            for e in self.evaluations.values()
                            if e["project"] == project
                            and (params.get("run_id") is None or e["training_run_id"] == params["run_id"])
                            and (params.get("model") is None or e["model"] == params["model"])
                        ),
                        key=lambda e: e["created_at"],
                        reverse=True,
                    )
                    return 200, self.page(items, params)
        # runs
        if len(p) >= 2 and p[0] == "runs":
            run = self.run(p[1])
            tail = p[2:]
            if not tail and m == "GET":
                return 200, self.run_view(run)
            if tail == ["forward"] or tail == ["forward_backward"]:
                return 202, self.forward(run, body, tail[0])
            if tail == ["optim_step"]:
                return 202, self.optim_step(run, body)
            if tail == ["save_checkpoint"]:
                return 202, self.save_checkpoint(run, body)
            if tail == ["close"]:
                if run["status"] not in ("closed", "failed"):
                    run["status"] = "closed"
                return 200, self.run_view(run)
            if tail == ["metrics"] and m == "POST":
                if len(body["metrics"]) > 100:
                    raise HTTPError(422, "invalid_input", "at most 100 keys", "metrics")
                step = body.get("step", run["step"])  # no step: the run's current step
                self.metrics.setdefault(run["id"], []).extend(
                    {"step": step, "key": k, "value": v, "source": "client", "time": now().isoformat()}
                    for k, v in body["metrics"].items()
                )
                return 204, None
            if tail == ["metrics"] and m == "GET":
                wanted = set(params["keys"].split(",")) if params.get("keys") else None
                since = int(params.get("since_step", 0))
                return 200, [
                    x
                    for x in self.metrics.get(run["id"], [])
                    if x["step"] >= since and (not wanted or x["key"] in wanted)
                ]
            if tail == ["evaluations"] and m == "POST":
                e = {
                    "id": self.new_id("evl"),
                    "project": run["project"],
                    "model": body["model"],
                    "dataset": None,
                    "training_run_id": run["id"],
                    "source": "client",
                    "status": "completed",
                    "results": {"name": body.get("name", "eval"), "step": body.get("step"), **body["results"]},
                    "created_at": now().isoformat(),
                }
                self.evaluations[e["id"]] = e
                return 201, e
        # futures
        if p[:1] == ["futures"]:
            if p[1:] == ["retrieve"] and m == "POST":
                ids = body["ids"]
                if not 1 <= len(ids) <= 256:
                    raise HTTPError(422, "invalid_input", "1..256 ids", "ids")
                for i in ids:
                    if i not in self.futures:
                        raise HTTPError(404, "not_found", f"no future {i}")
                return 200, [self.poll(self.futures[i]) for i in ids]
            if len(p) == 2 and m == "GET":
                if p[1] not in self.futures:
                    raise HTTPError(404, "not_found", f"no future {p[1]}")
                return 200, self.poll(self.futures[p[1]])
            if len(p) == 3 and p[2] == "cancel" and m == "POST":
                f = self.futures.get(p[1])
                if f is None:
                    raise HTTPError(404, "not_found", f"no future {p[1]}")
                if f["status"] in ("pending", "running"):
                    f["status"], f["_polls"] = "cancelled", 0
                    f["error"] = {"code": "cancelled", "message": "cancelled by the client"}
                return 200, {k: v for k, v in f.items() if not k.startswith("_")}
        if p[:1] == ["evaluations"] and len(p) == 2 and m == "GET":
            if p[1] not in self.evaluations:
                raise HTTPError(404, "not_found", f"no evaluation {p[1]}")
            return 200, self.evaluations[p[1]]
        raise HTTPError(404, "not_found", f"no route {method} {path}")

    @staticmethod
    def page(items: list, params: dict[str, str]) -> dict:
        limit, offset = int(params.get("limit", 50)), int(params.get("offset", 0))
        return {"items": items[offset : offset + limit], "total": len(items)}

    # ------------------------------------------------------------------ decisions
    def resolve_model(self, model: str, project: str | None = None) -> tuple[str, str]:
        """(base model id, the id of the model that answers) for ``<project>``, ``<project>/base`` or
        ``<project>/<name>``; with ``project``, also a bare name in it (``base`` included). As the API's
        services/models.resolve."""
        if model in BASE_MODELS:
            raise HTTPError(
                422,
                "model_requires_project",
                f'Call {model} through a project: "<project>/base" for the base model, or "<project>" for the '
                "project's live model.",
                "model",
            )
        if "/" in model:
            proj, _, name = model.partition("/")
        elif project is not None and model != project:
            proj, name = project, model
        else:
            proj, name = model, ""
        p = self.projects.get(proj)
        if p is None:
            raise HTTPError(404, "unknown_model", f"no model {model}")
        if not name:
            name = (self.live_model(p) or f"{proj}/{BASE}").partition("/")[2]
        if name == BASE:
            if p["base_model"] is None:
                raise HTTPError(409, "no_base_model", f"Project {proj} has no base model yet.", "model")
            return p["base_model"], f"{proj}/{BASE}"
        mid = f"{proj}/{name}"
        if mid not in self.models or not self.models[mid]["_ready"]:
            raise HTTPError(404, "unknown_model", f"no model {model}")
        return self.models[mid]["base_model"], mid

    def systemone(self, body: dict) -> dict:
        for k in ("model", "state", "questions"):
            if k not in body:
                raise HTTPError(422, "invalid_input", f"missing {k}", k)
        if not 1 <= len(body["questions"]) <= 64:
            raise HTTPError(422, "invalid_input", "send 1..64 questions", "questions")
        base_id, answered = self.resolve_model(body["model"])
        base = BASE_MODELS[base_id]
        self.check_balance()
        answers = {}
        for name, q in body["questions"].items():
            check_question(q, f"questions.{name}")
            if len(option_keys(q)) > base["max_options"]:
                raise HTTPError(422, "invalid_options", f"questions.{name}: more than {base['max_options']} options")
            answers[name] = answer_from_probs(q, uniform(q))
        return {
            "model": answered,
            "answers": answers,
            "usage": {"input_tokens": 7 * len(answers), "output_tokens": 0},
        }

    def list_models(self) -> dict:
        """GET /v1/models: the names a decision can send (`<project>`, `<project>/base`, `<project>/<name>`)."""
        items: list[dict] = []
        for name, p in sorted(self.projects.items()):
            if p["base_model"] is None:
                continue
            serves = self.live_model(p) or f"{name}/{BASE}"
            items.append(
                {
                    "name": name,
                    "description": f"{name}: the project's live model, now {serves}.",
                    "release_date": "2026-10-01",
                    "endor": {"kind": "live", "project": name, "base_model": p["base_model"], "model": serves},
                }
            )
            items.append(
                {
                    "name": f"{name}/{BASE}",
                    "description": f"{name}: the base model {p['base_model']}.",
                    "release_date": "2026-10-01",
                    "endor": {"kind": "base", "project": name, "base_model": p["base_model"], "contract": "c"},
                }
            )
        items += [
            {
                "name": m["id"],
                "description": f"{m['project']}: fine-tuned {m['base_model']}",
                "release_date": m["created_at"][:10],
                "endor": {
                    "kind": "model",
                    "project": m["project"],
                    "base_model": m["base_model"],
                    "training_run_id": m["training_run_id"],
                    "step": m["step"],
                    "parent_model": m["parent_model"],
                },
            }
            for m in self.models.values()
            if m["_ready"]
        ]
        return {"models": items}

    def list_base_models(self) -> dict:
        """GET /v1/base_models: the catalog."""
        return {
            "base_models": [
                {
                    "id": k,
                    "params": "1B",
                    "hf_repo": f"org/{k}",
                    "hf_revision": "abc123",
                    "contract": "c",
                    "contract_version": 1,
                    "max_rank": 64,
                    "trainer_gpu": "H100",
                    **v,
                    "price_per_mtok_decide": 0.5,
                    "price_per_mtok_decide_continuous_learning": 0.75,
                    "price_per_gpu_hour": 3.0,
                }
                for k, v in BASE_MODELS.items()
            ]
        }

    # ------------------------------------------------------------------ projects, datasets
    def create_project(self, body: dict) -> dict:
        name = body.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            raise HTTPError(422, "invalid_input", "bad name", "name")
        if name in self.projects:
            raise HTTPError(409, "conflict", f"project {name} already exists")
        cap = int(LIMITS["max_projects_per_org"])
        if len(self.projects) >= cap:
            raise HTTPError(
                409,
                "limit_reached",
                f"This organization has {len(self.projects)} projects, the maximum of {cap}. Delete one first.",
            )
        kind = body.get("kind", "custom")
        if kind not in ("custom", "managed"):
            raise HTTPError(422, "invalid_input", "kind is custom or managed", "kind")
        if body.get("base_model") not in BASE_MODELS:
            raise HTTPError(422, "unknown_model", f"Not a base model: {body.get('base_model')!r}.", "base_model")
        p = {
            "name": name,
            "description": body.get("description"),
            "kind": kind,
            "base_model": body["base_model"],
            "live_model": None,
            "paused": False,
            "created_at": now().isoformat(),
        }
        self.projects[name] = p
        return self.project_view(p, self)

    def datasets_route(self, m: str, project: str, tail: list[str], body: Any, params: dict) -> tuple[int, Any]:
        if not tail and m == "POST":
            name, rows = body.get("name"), body.get("rows")
            if not isinstance(name, str) or not NAME.match(name):
                raise HTTPError(422, "invalid_input", "bad name", "name")
            if not isinstance(rows, list) or not 1 <= len(rows) <= 50_000:
                raise HTTPError(422, "invalid_input", "1..50000 rows", "rows")
            if (project, name) in self.datasets:
                raise HTTPError(409, "conflict", f"dataset {name} already exists")
            n_labelled, types = 0, Counter()
            for i, row in enumerate(rows):
                try:
                    if not row.get("questions"):
                        raise ValueError("a row needs at least one question")
                    for qn, q in row["questions"].items():
                        check_question(q, f"rows[{i}].questions.{qn}")
                        types[q["type"]] += 1
                    for qn, label in (row.get("labels") or {}).items():
                        if qn not in row["questions"]:
                            raise ValueError(f"label for unknown question {qn}")
                        label_target(row["questions"][qn], label)
                        n_labelled += 1
                except ValueError as e:
                    raise HTTPError(422, "invalid_row", f"rows[{i}]: {e}") from e
            d = {
                "name": name,
                "project": project,
                "n_rows": len(rows),
                "n_labelled_questions": n_labelled,
                "question_types": dict(types),
                "created_at": now().isoformat(),
                "_rows": rows,
            }
            self.datasets[(project, name)] = d
            return 201, {k: v for k, v in d.items() if not k.startswith("_")}
        if not tail and m == "GET":
            items = [
                {k: v for k, v in d.items() if not k.startswith("_")}
                for (p, _), d in self.datasets.items()
                if p == project
            ]
            return 200, self.page(items, params)
        if tail and (project, tail[0]) not in self.datasets:
            raise HTTPError(404, "not_found", f"no dataset {tail[0]}")
        d = self.datasets[(project, tail[0])]
        if len(tail) == 1 and m == "GET":
            return 200, {k: v for k, v in d.items() if not k.startswith("_")}
        if len(tail) == 1 and m == "DELETE":
            del self.datasets[(project, tail[0])]
            return 204, None
        if tail[1:] == ["rows"] and m == "GET":
            if int(params.get("limit", 50)) > 1000:
                raise HTTPError(422, "invalid_input", "limit <= 1000", "limit")
            return 200, self.page(d["_rows"], params)
        raise HTTPError(404, "not_found", "no route")

    # ------------------------------------------------------------------ runs
    def check_balance(self) -> None:
        if self.blocked:
            raise HTTPError(402, "insufficient_balance", "Your balance is used up. Add credits in the dashboard.")

    def create_run(self, project: str, body: dict) -> dict:
        self.check_balance()
        open_runs = [r["id"] for r in self.runs.values() if r["status"] not in ("closed", "failed")]
        if len(open_runs) >= MAX_ACTIVE_RUNS:
            raise HTTPError(
                409,
                "limit_reached",
                f"This organization has {len(open_runs)} open runs, the maximum of {MAX_ACTIVE_RUNS} (idle ones "
                f"count): {', '.join(open_runs)}. Close one first: run.close() in the SDK or "
                "`endor runs close <run_id>`.",
            )
        proj = self.project(project)
        if body.get("base_model") is not None and body["base_model"] != proj["base_model"]:
            raise HTTPError(
                422,
                "invalid_input",
                f"Project {project} trains on its base model {proj['base_model']}, not {body['base_model']}.",
                "base_model",
            )
        base_model, parent = proj["base_model"], None
        if body.get("from_model"):
            mid = body["from_model"] if "/" in body["from_model"] else f"{project}/{body['from_model']}"
            m = self.models.get(mid)
            if m is None:
                raise HTTPError(
                    422, "unknown_model", f"No model {body['from_model']!r} in project {project!r}.", "from_model"
                )
            if m["project"] != project:
                raise HTTPError(422, "invalid_input", "from_model must belong to this project", "from_model")
            if body.get("include_optimizer") and not m["has_optimizer"]:
                raise HTTPError(409, "invalid_state", f"{mid} was saved without optimizer state")
            base_model, parent = m["base_model"], mid
        if base_model not in BASE_MODELS:
            raise HTTPError(422, "unknown_model", f"not a trainable base model: {base_model}", "base_model")
        defaults = {
            "rank": 16,
            "alpha": 32.0,
            "seed": None,
            "train_attn": True,
            "train_mlp": True,
            "train_readout": False,
        }
        if parent:  # a run from a saved model inherits its LoRA settings; explicit conflicting values are a 422
            inherited = self.runs[self.models[parent]["training_run_id"]]["lora"]
            for k, v in (body.get("lora") or {}).items():
                if k != "seed" and v != inherited[k]:
                    raise HTTPError(422, "invalid_input", f"{parent} was trained with {k}={inherited[k]}", f"lora.{k}")
            defaults = inherited
        lora = {**defaults, **(body.get("lora") or {})}
        if not 1 <= lora["rank"] <= 256:
            raise HTTPError(422, "invalid_input", "rank 1..256", "lora.rank")
        run_id = self.new_id("run")
        ready = self.future("provision_run", {"run_id": run_id}, pending=self.provisioning_polls)
        run = {
            "id": run_id,
            "project": project,
            "name": body.get("name"),
            "base_model": base_model,
            "contract": "letters-16",
            "lora": lora,
            "status": "provisioning" if self.provisioning_polls else "ready",
            "ready_future_id": ready["id"],
            "step": 0,
            "next_seq_id": 0,
            "parent_model": parent,
            "tags": body.get("tags") or [],
            "config": body.get("config") or {},
            "code_hash": body.get("code_hash"),
            "user_metadata": body.get("user_metadata") or {},
            "failure": None,
            "created_at": now().isoformat(),
            "_w": 0.0,
        }
        self.runs[run_id] = run
        return self.run_view(run)

    def accept(self, run: dict, body: dict, kind: str, make_result) -> dict:
        """The sequence-number rule: equal -> accepted; below with the same body -> idempotent; else 409."""
        if run["status"] in ("closed", "failed"):
            raise HTTPError(409, "invalid_state", f"run is {run['status']}")
        if "seq_id" not in body:
            raise HTTPError(422, "invalid_input", "seq_id is required", "seq_id")
        seq = body["seq_id"]
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if seq < run["next_seq_id"]:
            prev = self.ops.get((run["id"], seq))
            if prev and prev[0] == digest:
                return {"future_id": prev[1]}
            raise HTTPError(409, "seq_conflict", f"expected seq_id {run['next_seq_id']}, got {seq}", "seq_id")
        if seq != run["next_seq_id"]:
            raise HTTPError(409, "seq_conflict", f"expected seq_id {run['next_seq_id']}, got {seq}", "seq_id")
        self.check_balance()
        if run["status"] == "provisioning" and self.poll(self.futures[run["ready_future_id"]])["status"] == "completed":
            run["status"] = "ready"
        result = make_result()
        f = self.future(kind, error=result["_error"]) if "_error" in result else self.future(kind, result)
        run["next_seq_id"] = seq + 1
        self.ops[(run["id"], seq)] = (digest, f["id"])
        return {"future_id": f["id"]}

    def forward(self, run: dict, body: dict, kind: str) -> dict:
        data = body.get("data")
        if not isinstance(data, list) or not 1 <= len(data) <= 1024:
            raise HTTPError(422, "invalid_input", "1..1024 datums", "data")
        if body.get("loss_fn", "cross_entropy") not in ("cross_entropy", "brier"):
            raise HTTPError(422, "invalid_input", "bad loss_fn", "loss_fn")
        base = BASE_MODELS[run["base_model"]]
        for i, d in enumerate(data):
            check_datum(d, base, f"data[{i}]")

        def result() -> dict:
            outs, loss_sum, w_sum, correct = [], 0.0, 0.0, 0
            for d in data:
                keys = option_keys(d["question"])
                probs = dict(zip(keys, uniform(d["question"]), strict=False))
                t = d["target"]
                y = {t["label"]: 1.0} if "label" in t else t["probs"]
                loss = -sum(w * math.log(probs[k]) for k, w in y.items() if w > 0)
                w = d.get("weight", 1.0)
                outs.append({"probabilities": probs, "loss": loss})
                loss_sum += w * loss
                w_sum += w
                correct += int(max(probs, key=probs.get) == max(y, key=y.get))
            n = len(data)
            metrics = {
                "loss:sum": loss_sum,
                "weight:sum": w_sum,
                "loss:mean": loss_sum / w_sum if w_sum else 0.0,
                "accuracy": correct / n,
                "n": n,
            }
            if kind == "forward_backward":
                run["_w"] += w_sum
                self.auto_metrics(run, {"train/loss": metrics["loss:mean"], "train/accuracy": metrics["accuracy"]})
            return {"outputs": outs, "metrics": metrics}

        return self.accept(run, body, kind, result)

    def optim_step(self, run: dict, body: dict) -> dict:
        adam = body.get("adam_params") or {}

        def result() -> dict:
            if run["_w"] <= 0:
                return {
                    "_error": {
                        "code": "no_gradients",
                        "message": "Nothing to apply: no forward_backward with "
                        "a positive weight since the last optim_step.",
                    }
                }
            run["_w"] = 0.0
            run["step"] += 1
            lr = adam.get("learning_rate", 1e-4)
            self.auto_metrics(run, {"train/grad_norm": 0.0, "train/lr": lr})
            return {"step": run["step"], "grad_norm": 0.0, "learning_rate": lr}

        return self.accept(run, body, "optim_step", result)

    def auto_metrics(self, run: dict, values: dict[str, float]) -> None:
        self.metrics.setdefault(run["id"], []).extend(
            {"step": run["step"], "key": k, "value": v, "source": "auto", "time": now().isoformat()}
            for k, v in values.items()
        )

    def save_checkpoint(self, run: dict, body: dict) -> dict:
        name = body.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            raise HTTPError(422, "invalid_input", "bad name", "name")
        if name == BASE:  # as the API's models.check_sdk_name
            raise HTTPError(422, "invalid_input", '"base" is reserved for the project\'s base model.', "name")
        ttl = body.get("ttl_seconds")
        if ttl is not None and not 3600 <= ttl <= 10 * 365 * 86400:
            raise HTTPError(422, "invalid_input", "ttl 1h..10y", "ttl_seconds")
        mid = f"{run['project']}/{name}"
        if "seq_id" in body and body["seq_id"] >= run["next_seq_id"] and mid in self.models:
            raise HTTPError(409, "conflict", f"model {mid} already exists")
        n, cap = (
            sum(1 for m in self.models.values() if m["project"] == run["project"]),
            LIMITS["max_models_per_project"],
        )
        if "seq_id" in body and body["seq_id"] >= run["next_seq_id"] and n >= cap:
            raise HTTPError(409, "limit_reached", f"Project {run['project']} has {n} models, the maximum.", "name")

        def result() -> dict:
            self.models[mid] = {
                "id": mid,
                "project": run["project"],
                "name": name,
                "training_run_id": run["id"],
                "base_model": run["base_model"],
                "contract": "letters-16",
                "parent_model": run["parent_model"],
                "step": run["step"],
                "has_optimizer": bool(body.get("include_optimizer")),
                "size_bytes": 1024,
                "expires_at": (now() + timedelta(seconds=ttl)).isoformat() if ttl else None,
                "user_metadata": body.get("user_metadata") or {},
                "created_at": now().isoformat(),
                "_ready": True,
            }
            return {"model": mid}

        return self.accept(run, body, "save_checkpoint", result)

    # ------------------------------------------------------------------ models
    def models_route(self, m: str, project: str, tail: list[str], body: Any, params: dict) -> tuple[int, Any]:
        proj = self.project(project)
        has_base = proj["base_model"] is not None
        if not tail and m == "GET":
            base = [self.base_model_view(proj)] if has_base and params.get("run_id") is None else []
            items = sorted(
                (
                    x
                    for x in self.models.values()
                    if x["project"] == project
                    and x["_ready"]
                    and (params.get("run_id") is None or x["training_run_id"] == params["run_id"])
                ),
                key=lambda x: x["created_at"],
                reverse=True,
            )
            return 200, self.page(base + [self.model_view(x) for x in items], params)
        if tail[0] == BASE and has_base and len(tail) == 1 and m == "GET":
            return 200, self.base_model_view(proj)
        if tail[0] == BASE and len(tail) == 1 and m == "DELETE":
            raise HTTPError(422, "invalid_input", "The project's base model can't be deleted.", "name")
        mid = f"{project}/{tail[0]}"
        if mid not in self.models:
            raise HTTPError(404, "not_found", f"no model {mid}")
        mod = self.models[mid]
        if len(tail) == 1 and m == "GET":
            return 200, self.model_view(mod)
        if len(tail) == 1 and m == "PATCH":
            if "ttl_seconds" in body:
                ttl = body["ttl_seconds"]
                if ttl is not None and not 3600 <= ttl <= 10 * 365 * 86400:
                    raise HTTPError(422, "invalid_input", "ttl 1h..10y", "ttl_seconds")
                mod["expires_at"] = (now() + timedelta(seconds=ttl)).isoformat() if ttl else None
            return 200, self.model_view(mod)
        if len(tail) == 1 and m == "DELETE":
            del self.models[mid]
            if proj["live_model"] == mid:
                proj["live_model"] = None
            return 204, None
        if tail[1:] == ["download"] and m == "GET":
            names = ["adapter_model.safetensors", "adapter_config.json", "endor_manifest.json"]
            if mod["has_optimizer"] and params.get("include_optimizer") == "true":
                names.append("optimizer.pt")
            files = []
            for n in names:
                data = self.file_bytes(mid, n)
                token = uuid.uuid4().hex
                self.file_links[token] = (mid, n)
                files.append(
                    {
                        "name": n,
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "url": f"https://{FILES_HOST}/{token}/{n}?X-Amz-Signature=sig{token}&X-Amz-Expires=600",
                    }
                )
            return 200, {"files": files, "expires_at": (now() + timedelta(minutes=10)).isoformat()}
        raise HTTPError(404, "not_found", "no route")

    # ------------------------------------------------------------------ evaluations
    def create_evaluation(self, project: str, body: dict) -> dict:
        self.check_balance()
        if (project, body.get("dataset")) not in self.datasets:
            raise HTTPError(404, "not_found", f"no dataset {body.get('dataset')}")
        self.resolve_model(body["model"], project)
        e = {
            "id": self.new_id("evl"),
            "project": project,
            "model": body["model"],
            "dataset": body["dataset"],
            "training_run_id": body.get("training_run_id"),
            "source": "server",
            "status": "completed",
            "results": {"n": 0, "overall": {"accuracy": 0.5}},
            "created_at": now().isoformat(),
        }
        self.evaluations[e["id"]] = e
        return {"future_id": self.future("evaluate", {"evaluation_id": e["id"]})["id"]}
