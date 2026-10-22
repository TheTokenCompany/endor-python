"""An in-memory fake of the Endor API, served through ``httpx.MockTransport``.

It implements the public contract the SDK relies on: bearer auth, the error envelope, strict sequence numbers with
idempotent retries, ``Idempotency-Key`` on creates, long-polled futures (optionally pending for N polls), the
decision answer formulas, the model references (``<project>``, ``<project>:preview``, a production custom model's
name), continuous learning projects with a production version and a preview (the newest version while it isn't in
production), custom models in production or archived, the count limits (409 ``limit_reached``, ``production_full``),
``no_gradients``, 402 for a blocked
org, and fault injection for retry tests.
It is a test double, not a reference server.
"""

from __future__ import annotations

import base64
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
UPLOADS_HOST = "uploads.endor.test"  # where declared images are PUT (the API's presigned links)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES_PER_CALL = 128
NOT_ENABLED = (
    "Fine-tuning with images is in limited access and isn't enabled for this organization yet. To request access, "
    "write to support@thetokencompany.com with your organization id (GET /v1/whoami)."
)
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")

LIMITS: dict[str, float] = {
    "max_projects_per_org": 5,
    "max_production_models_per_org": 3,
    "max_custom_models_per_org": 500,
    "max_datasets_per_org": 100,
    "max_api_keys_per_org": 20,
    "max_active_runs": 4,
    "max_dataset_rows": 50_000,
    "max_pending_ops": 64,
    "decisions_per_minute": 60,
    "downloads_per_minute": 30,
    "dataset_uploads_per_minute": 20,
    "runs_per_minute": 20,
    "evaluations_per_minute": 30,
}
MAX_ACTIVE_RUNS = int(LIMITS["max_active_runs"])
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
        "modalities": ["text", "image"],
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


IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")


def find_images(v: Any, path: str = "state") -> list[tuple[str, dict]]:
    """The images in a state, inline ({"type": "image", ..., "data": ...}) or references ({"type": "image",
    "sha256"}), with their paths, as the API reads them."""
    if isinstance(v, dict) and v.get("type") == "image" and ("data" in v or "sha256" in v):
        return [(path, v)]
    if isinstance(v, dict):
        return [x for k, item in v.items() for x in find_images(item, f"{path}.{k}")]
    if isinstance(v, list):
        return [x for i, item in enumerate(v) for x in find_images(item, f"{path}.{i}")]
    return []


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
    heartbeat_fault: Exception | None = None
    """Raised by every run heartbeat while set: an ``HTTPError`` or an ``httpx.TransportError``."""
    requests: list[Recorded] = field(default_factory=list)
    file_requests: list[Recorded] = field(default_factory=list)  # GETs of presigned file links
    file_links: dict[str, tuple[str, str]] = field(default_factory=dict)  # token -> (model id, file name)
    blobs: dict[str, bytes] = field(default_factory=dict)  # token -> an evaluation's rows.jsonl
    corrupt_files: bool = False
    blocked: bool = False
    """The org's balance is used up: paid calls answer 402 insufficient_balance."""
    balance_usd: float | None = 100.0
    """What GET /v1/balance answers (None: a server without billing)."""
    image_training: bool = True
    """Whether the org may fine-tune with images (403 image_training_not_enabled otherwise)."""
    continuous_learning: bool = True
    """What GET /v1/whoami says about the org's continuous learning switch."""
    images: dict[str, dict] = field(default_factory=dict)  # sha256 -> {media_type, bytes, status}
    uploads: dict[str, bytes] = field(default_factory=dict)  # sha256 -> the bytes PUT to its link
    upload_requests: list[Recorded] = field(default_factory=list)  # PUTs to the upload links
    fail_uploads: int = 0
    """How many of the next PUTs to upload links answer 500."""

    projects: dict[str, dict] = field(default_factory=dict)  # continuous learning projects, by name
    versions: dict[str, list[dict]] = field(default_factory=dict)  # project -> its versions, oldest first
    wandb_connected: bool = False  # the org's W&B connection (dashboard only)
    user_reports: list[dict] = field(default_factory=list)  # POST /v1/user_reports, with "org_id"
    decisions: dict[str, dict] = field(default_factory=dict)  # request id -> {"project", "answers"} of a decision
    corrections: dict[str, dict] = field(default_factory=dict)  # request id -> its active correction
    learning: dict[str, dict] = field(default_factory=dict)  # project -> fields its GET .../learning answers with
    request_id: str = ""  # the current request's X-Request-Id
    datasets: dict[str, dict] = field(default_factory=dict)  # the org's datasets, by name
    runs: dict[str, dict] = field(default_factory=dict)
    models: dict[str, dict] = field(default_factory=dict)  # the org's custom models, by name
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

    def preview(self, p: dict) -> str | None:
        """The project's preview: its newest version while that isn't live (so never while auto-promote is on)."""
        vs = self.versions.get(p["name"], [])
        return vs[-1]["name"] if vs and vs[-1]["name"] != p["live"] else None

    def project_view(self, p: dict) -> dict:
        vs = self.versions.get(p["name"], [])
        return {
            "name": p["name"],
            "description": p["description"],
            "base_model": p["base_model"],
            "production": p["live"],
            "preview": self.preview(p),
            "auto_promote": p["auto_promote"],
            "promoted_at": p["promoted_at"],
            "versions": len(vs),
            "created_at": p["created_at"],
        }

    def version_view(self, p: dict, v: dict) -> dict:
        return {
            **v,
            "base_model": p["base_model"],
            "production": v["name"] == p["live"],
            "preview": v["name"] == self.preview(p),
        }

    def add_version(self, project: str, accuracy: float | None = None, loss: float | None = None) -> str:
        """What learning does when it keeps a version: the next ``v<n>`` goes live while auto-promote is on, else it is
        the preview. Returns its name."""
        p, vs = self.project(project), self.versions.setdefault(project, [])
        v = {
            "name": f"v{len(vs) + 1}",
            "accuracy": accuracy,
            "loss": loss,
            "size_bytes": 1024,
            "created_at": now().isoformat(),
        }
        vs.append(v)
        if p["auto_promote"]:
            p["live"], p["promoted_at"] = v["name"], now().isoformat()
        return v["name"]

    def loss_at(self, m: dict) -> float | None:
        """The run's train/loss at the model's step, or the last one before it (the mean of that step's points)."""
        points = [
            x
            for x in self.metrics.get(m.get("training_run_id") or "", [])
            if x["key"] == "train/loss" and x["step"] <= m["step"]
        ]
        if not points:
            return None
        last = max(x["step"] for x in points)
        at = [x["value"] for x in points if x["step"] == last]
        return sum(at) / len(at)

    def project(self, name: str) -> dict:
        if name not in self.projects:
            raise HTTPError(404, "not_found", f"no project {name}")
        return self.projects[name]

    def custom_model(self, name: str) -> dict:
        m = self.models.get(name)
        if m is None or not m["_ready"]:
            raise HTTPError(404, "not_found", f"No custom model {name!r}.")
        return m

    def name_taken(self, name: str) -> bool:
        """One namespace per org: projects and custom models, and the base model ids."""
        return name in self.projects or name in self.models or name in BASE_MODELS

    def run(self, run_id: str) -> dict:
        if run_id not in self.runs:
            raise HTTPError(404, "not_found", f"no run {run_id}")
        return self.runs[run_id]

    @staticmethod
    def run_view(r: dict) -> dict:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    def model_view(self, m: dict) -> dict:
        """A custom model as the API shows it (`live_at` is the fake's own field, as in the API's table). A promoted
        model is `serving` live at once; set its `_serving` (and `_serving_error`) to show another state."""
        live = m["live_at"] is not None
        serving = m.get("_serving", "live") if live else None
        return {
            **{k: v for k, v in m.items() if not k.startswith("_") and k != "live_at"},
            "production": live,
            "state": "production" if live else "archived",
            "promoted_at": m["live_at"],
            "serving": serving,
            "serving_error": m.get("_serving_error") if serving == "failed" else None,
            "loss": self.loss_at(m),
        }

    @staticmethod
    def file_bytes(mid: str, name: str) -> bytes:
        return f"{mid}:{name}:".encode() * 1000

    def download_files(self, key: str, has_optimizer: bool, include_optimizer: bool) -> dict:
        names = ["adapter_model.safetensors", "adapter_config.json", "endor_manifest.json"]
        if has_optimizer and include_optimizer:
            names.append("optimizer.pt")
        files = []
        for n in names:
            data = self.file_bytes(key, n)
            token = uuid.uuid4().hex
            self.file_links[token] = (key, n)
            files.append(
                {
                    "name": n,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "url": f"https://{FILES_HOST}/{token}/{n}?X-Amz-Signature=sig{token}&X-Amz-Expires=600",
                }
            )
        return {"files": files, "expires_at": (now() + timedelta(minutes=10)).isoformat()}

    def store_upload(self, request: httpx.Request) -> httpx.Response:
        """A PUT to an image's upload link: S3's rules, no Endor auth. The link is bound to the hash: the checksum
        header must be the declared hash (else 403, as a signature mismatch) and the bytes must have it (else 400
        BadDigest)."""
        sha = request.url.path.strip("/")
        if self.fail_uploads > 0:
            self.fail_uploads -= 1
            return httpx.Response(500, text="<Error><Code>InternalError</Code></Error>")
        if request.method != "PUT" or sha not in self.images:
            return httpx.Response(403, text="<Error><Code>AccessDenied</Code></Error>")
        want = base64.b64encode(bytes.fromhex(sha)).decode()
        if request.headers.get("x-amz-checksum-sha256") != want or "content-type" not in request.headers:
            return httpx.Response(403, text="<Error><Code>SignatureDoesNotMatch</Code></Error>")
        if "authorization" in request.headers or "x-endor-api-key" in request.headers:
            return httpx.Response(
                403, text="<Error><Code>AccessDenied</Code>the API key must not reach storage</Error>"
            )
        if hashlib.sha256(request.content).hexdigest() != sha:
            return httpx.Response(400, text="<Error><Code>BadDigest</Code></Error>")
        self.uploads[sha] = request.content
        return httpx.Response(200, headers={"etag": f'"{sha[:32]}"'})

    def declare_images(self, body: dict) -> dict:
        """POST /v1/images as the API answers it: per declared image, ready, an upload link, or rejected."""
        if not self.image_training:
            raise HTTPError(403, "image_training_not_enabled", NOT_ENABLED)
        items = body.get("images")
        if not isinstance(items, list) or not 1 <= len(items) <= 1000:
            raise HTTPError(422, "invalid_input", "1..1000 images", "images")
        out = []
        for i, it in enumerate(items):
            sha, mt, size = it.get("sha256"), it.get("media_type"), it.get("bytes")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise HTTPError(422, "invalid_input", "sha256: 64 lowercase hex characters", f"images.{i}.sha256")
            if mt not in IMAGE_TYPES:
                raise HTTPError(422, "invalid_input", "bad media_type", f"images.{i}.media_type")
            if not isinstance(size, int) or size < 1:
                raise HTTPError(422, "invalid_input", "bytes >= 1", f"images.{i}.bytes")
            if size > MAX_IMAGE_BYTES:
                raise HTTPError(422, "invalid_input", "the image is over 5 MB", f"images.{i}.bytes")
            row = self.images.setdefault(sha, {"media_type": mt, "bytes": size, "status": "pending"})
            if row["media_type"] != mt:
                raise HTTPError(422, "invalid_input", f"declared as {row['media_type']}", f"images.{i}.media_type")
            base = {"sha256": sha, "media_type": mt, "bytes": row["bytes"]}
            if row["status"] == "ready":
                out.append({**base, "status": "ready"})
            elif sha in self.uploads:
                data = self.uploads[sha]
                magic = {
                    "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
                    "image/jpeg": data.startswith(b"\xff\xd8\xff"),
                    "image/webp": data[:4] == b"RIFF" and data[8:12] == b"WEBP",
                }[mt]
                if len(data) > MAX_IMAGE_BYTES or not magic:
                    del self.uploads[sha], self.images[sha]
                    reason = "the image is over 5 MB" if len(data) > MAX_IMAGE_BYTES else f"not {mt}"
                    out.append({**base, "status": "rejected", "reason": reason})
                else:
                    row["status"], row["bytes"] = "ready", len(data)
                    out.append({**base, "bytes": len(data), "status": "ready"})
            else:
                out.append(
                    {
                        **base,
                        "status": "upload",
                        "url": f"https://{UPLOADS_HOST}/{sha}?X-Amz-Signature=sig{sha[:8]}&X-Amz-Expires=3600",
                        "headers": {
                            "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(sha)).decode(),
                            "content-type": mt,
                        },
                        "expires_at": (now() + timedelta(hours=1)).isoformat(),
                    }
                )
        return {"images": out}

    def check_images(self, holders: list[dict], where: str, base: dict | None = None) -> int:
        """The images in these states (rows' or datums'), as the API checks training data: access, references to
        ready images only (the SDK uploads inline images first, so inline here is a bug), a base that reads them;
        how many there are."""
        n = 0
        for i, h in enumerate(holders):
            found = find_images(h.get("state"))
            if not found:
                continue
            if not self.image_training:
                raise HTTPError(403, "image_training_not_enabled", NOT_ENABLED)
            if base is not None and "image" not in base.get("modalities", ["text"]):
                raise HTTPError(422, "unsupported_modality", "this base reads text only", f"{where}{i}.{found[0][0]}")
            if len(found) > 8:
                raise HTTPError(422, "invalid_input", "at most 8 images", f"{where}{i}.state")
            for path, im in found:
                full = f"{where}{i}.{path}"
                if "data" in im:
                    raise HTTPError(422, "invalid_input", "the SDK uploads images and sends references", full)
                row = self.images.get(im.get("sha256", ""))
                if row is None or row["status"] != "ready":
                    raise HTTPError(422, "invalid_input", "no uploaded image with this sha256", full)
                if im.get("media_type") not in (None, row["media_type"]):
                    raise HTTPError(422, "invalid_input", f"the uploaded image is {row['media_type']}", full)
            n += len(found)
        return n

    def serve_file(self, request: httpx.Request) -> httpx.Response:
        """A presigned storage GET: no Endor auth, the token in the path is the whole credential."""
        token, _, name = request.url.path.strip("/").partition("/")
        if token in self.blobs and name == "rows.jsonl" and request.method == "GET":
            return httpx.Response(200, content=self.blobs[token], headers={"content-type": "application/x-ndjson"})
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
        if request.url.host == UPLOADS_HOST:
            self.upload_requests.append(Recorded(request.method, request.url.path, dict(request.headers), None, {}))
            return self.store_upload(request)
        self.request_id = self.new_id("req")
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
            known = auth.startswith("Bearer ") and auth[7:] in self.api_keys
            if not known:
                raise HTTPError(401, "unauthorized", "bad API key")
            if request.method == "POST" and request.url.path == "/v1/user_reports":
                return self._json(201, self.user_report(body, "org_test"))
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
        headers = {"x-request-id": self.request_id}
        if replay:
            headers["idempotent-replayed"] = "true"
        if status == 204:
            return httpx.Response(204, headers=headers)
        return httpx.Response(status, json=payload, headers=headers)

    def _error(self, e: HTTPError) -> httpx.Response:
        err = {"code": e.code, "message": e.message}
        if e.param:
            err["param"] = e.param
        return httpx.Response(e.status, json={"error": err}, headers={"x-request-id": self.request_id, **e.headers})

    def user_report(self, body: Any, org_id: str) -> dict:
        if not isinstance(body, dict) or body.get("kind") not in ("bug", "confusing", "works"):
            raise HTTPError(422, "invalid_input", "kind must be bug, confusing or works", "kind")
        message = body.get("message")
        if not isinstance(message, str) or not 1 <= len(message) <= 4000:
            raise HTTPError(422, "invalid_input", "message must be 1 to 4000 characters", "message")
        context = body.get("context") or {}
        if len(context) > 30 or any(len(k) > 64 or (isinstance(v, str) and len(v) > 500) for k, v in context.items()):
            raise HTTPError(422, "invalid_input", "context is too big", "context")
        rep = {"id": self.new_id("urep"), "org_id": org_id, **body}
        self.user_reports.append(rep)
        return {"id": rep["id"], "message": "Thanks: the Endor team reads every report."}

    # ------------------------------------------------------------------ routing
    BODY_FIELDS: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: {
            ("POST", "projects"): {"name", "description", "base_model"},
            ("PATCH", "projects/*"): {"description", "auto_promote", "base_model"},
            ("POST", "runs"): {
                "base_model",
                "from_model",
                "model",
                "include_optimizer",
                "lora",
                "name",
                "tags",
                "config",
                "code_hash",
                "user_metadata",
                "total_steps",
                "wandb",
            },
            ("POST", "evaluations"): {"model", "dataset", "rows", "training_run_id", "metrics"},
            ("POST", "datasets"): {"name", "rows"},
            ("POST", "custom_models/*/promote"): {"replace"},
            ("PATCH", "custom_models/*"): {"ttl_seconds"},
            ("POST", "runs/*/save_checkpoint"): {"name", "include_optimizer", "ttl_seconds", "user_metadata", "seq_id"},
        }
    )
    """Request bodies the API checks strictly (an unknown field is 422 invalid_input naming it)."""

    def check_body(self, method: str, p: list[str], body: Any) -> None:
        shape = "/".join(x if i % 2 == 0 else "*" for i, x in enumerate(p))
        allowed = self.BODY_FIELDS.get((method, shape))
        if allowed is not None and isinstance(body, dict) and set(body) - allowed:
            extra = sorted(set(body) - allowed)[0]
            raise HTTPError(422, "invalid_input", f"{extra}: unknown field", extra)

    def route(self, method: str, path: str, body: Any, params: dict[str, str]) -> tuple[int, Any]:
        parts = [p for p in path.split("/") if p]
        if parts[:1] != ["v1"]:
            raise HTTPError(404, "not_found", "no such route")
        p = parts[1:]
        m = method
        self.check_body(m, p, body)
        if p == ["systemone"] and m == "POST":
            return 200, self.systemone(body)
        if p == ["corrections"] and m == "POST":
            return 200, self.keep_correction(self.check_correction(body, ""))
        if p == ["corrections", "batch"] and m == "POST":
            items = body.get("items") if isinstance(body, dict) else None
            if not isinstance(items, list) or not 1 <= len(items) <= 1000:
                raise HTTPError(422, "invalid_input", "send 1 to 1,000 items", "items")
            checked = [self.check_correction(it, f"items[{i}].") for i, it in enumerate(items)]
            return 200, {"items": [self.keep_correction(c) for c in checked]}
        if len(p) == 2 and p[0] == "corrections" and m in ("GET", "DELETE"):
            if p[1] not in self.decisions or p[1] not in self.corrections:
                raise HTTPError(404, "not_found", f"No correction of decision {p[1]!r}.")
            if m == "GET":
                return 200, self.corrections[p[1]]
            del self.corrections[p[1]]
            return 204, None
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
                "continuous_learning": self.continuous_learning,
                "image_training": self.image_training,
            }
        if p == ["images"] and m == "POST":
            return 200, self.declare_images(body)
        if p == ["images"] and m == "GET":
            items = [
                {"sha256": sha, **row, "created_at": now().isoformat(), "ready_at": None}
                for sha, row in self.images.items()
            ]
            return 200, self.page(items, params)
        if len(p) == 2 and p[0] == "images" and m == "GET":
            row = self.images.get(p[1])
            if row is None:
                raise HTTPError(404, "not_found", f"no image {p[1]}", "sha256")
            return 200, {"sha256": p[1], **row, "created_at": now().isoformat(), "ready_at": None}
        if p == ["balance"] and m == "GET":
            return 200, {"balance_usd": self.balance_usd}
        if p == ["usage"] and m == "GET":
            start, end = datetime.fromisoformat(params["starting_on"]), datetime.fromisoformat(params["ending_before"])
            if end - start > timedelta(days=14):
                raise HTTPError(422, "invalid_input", "at most 14 days per call", "ending_before")
            hour = start.replace(minute=0, second=0, microsecond=0).isoformat()
            common = {"hour": hour, "project_deleted": False, "base_model": "jev-9b"}
            return 200, [
                {
                    **common,
                    "kind": "decide",
                    "project": params.get("project"),
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
                    "project": None,
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
        # continuous learning projects
        if p == ["projects"] and m == "POST":
            return 201, self.create_project(body)
        if p == ["projects"] and m == "GET":
            items = sorted(self.projects.values(), key=lambda x: x["created_at"], reverse=True)
            return 200, self.page([self.project_view(x) for x in items], params)
        if len(p) == 2 and p[0] == "projects":
            proj = self.project(p[1])
            if m == "GET":
                return 200, self.project_view(proj)
            if m == "PATCH":
                if body.get("base_model") is not None or body.get("base_models") is not None:
                    raise HTTPError(422, "invalid_input", "A project's base model never changes.", "base_model")
                if body.get("paused") is not None:
                    raise HTTPError(422, "invalid_input", "Learning can't be paused.", "paused")
                if "description" in body:
                    proj["description"] = body["description"]
                if body.get("auto_promote") is not None:
                    proj["auto_promote"] = bool(body["auto_promote"])
                    vs = self.versions.get(p[1], [])
                    if proj["auto_promote"] and vs and proj["live"] != vs[-1]["name"]:
                        proj["live"], proj["promoted_at"] = vs[-1]["name"], now().isoformat()
                return 200, self.project_view(proj)
            if m == "DELETE":
                del self.projects[p[1]]
                self.versions.pop(p[1], None)
                return 204, None
        if len(p) >= 3 and p[0] == "projects":
            project, sub = p[1], p[2]
            proj = self.project(project)
            vs = self.versions.get(project, [])
            if sub == "learning" and len(p) == 3 and m == "GET":
                serving = f"{project}:{proj['live']}" if proj["live"] else project  # the base: plain "<project>"
                return 200, {
                    "status": "waiting",
                    "progress": 0.0,
                    "serving": serving,
                    "trained_at": None,
                    "trainings": [],
                    "drift": [],
                    **self.learning.get(project, {}),
                }
            if sub == "versions" and len(p) == 3 and m == "GET":
                return 200, self.page([self.version_view(proj, v) for v in reversed(vs)], params)
            if sub == "versions" and len(p) >= 4:
                v = next((x for x in vs if x["name"] == p[3]), None)
                if v is None:
                    raise HTTPError(404, "not_found", f"{project} has no version {p[3]!r}.", "version")
                if p[4:] == ["promote"] and m == "POST":
                    if v is not vs[-1]:  # an older version: a rollback stops auto-promote
                        proj["auto_promote"] = False
                    if proj["live"] != v["name"]:
                        proj["live"], proj["promoted_at"] = v["name"], now().isoformat()
                    return 200, self.project_view(proj)
                if p[4:] == ["download"] and m == "GET":
                    key = f"{project}:{v['name']}"
                    return 200, self.download_files(key, False, params.get("include_optimizer") == "true")
        # custom models
        if p[:1] == ["custom_models"]:
            return self.custom_models_route(m, p[1:], body, params)
        # datasets
        if p[:1] == ["datasets"]:
            return self.datasets_route(m, p[1:], body, params)
        # evaluations
        if p == ["evaluations"] and m == "POST":
            return 202, self.create_evaluation(body)
        if p == ["evaluations"] and m == "GET":
            items = sorted(
                (
                    e
                    for e in self.evaluations.values()
                    if (params.get("run_id") is None or e["training_run_id"] == params["run_id"])
                    and (params.get("model") is None or e["model"] == params["model"])
                ),
                key=lambda e: e["created_at"],
                reverse=True,
            )
            return 200, self.page([self.evaluation_view(e) for e in items], params)
        # runs
        if p == ["runs"] and m == "POST":
            return 201, self.create_run(body)
        if p == ["runs"] and m == "GET":
            items = sorted(
                (r for r in self.runs.values() if params.get("tag") is None or params["tag"] in r["tags"]),
                key=lambda r: r["created_at"],
                reverse=True,
            )
            return 200, self.page([self.run_view(r) for r in items], params)
        if len(p) >= 2 and p[0] == "runs":
            run = self.run(p[1])
            tail = p[2:]
            if not tail and m == "GET":
                return 200, self.run_view(run)
            if not tail and m == "PATCH":
                if "total_steps" in body:
                    t = body["total_steps"]
                    if t is not None and t < 1:
                        raise HTTPError(422, "invalid_input", "total_steps >= 1", "total_steps")
                    run["total_steps"] = t
                return 200, self.run_view(run)
            if tail == ["forward"] or tail == ["forward_backward"]:
                return 202, self.forward(run, body, tail[0])
            if tail == ["predict"]:
                return 202, self.predict(run, body)
            if tail == ["optim_step"]:
                return 202, self.optim_step(run, body)
            if tail == ["save_checkpoint"]:
                return 202, self.save_checkpoint(run, body)
            if tail == ["close"]:
                if run["status"] not in ("closed", "failed"):
                    run["status"] = "closed"
                return 200, self.run_view(run)
            if tail == ["heartbeat"] and m == "POST":
                if self.heartbeat_fault is not None:
                    raise self.heartbeat_fault
                if run["status"] in ("closed", "failed"):
                    raise HTTPError(409, "invalid_state", f"Run {run['id']} is closed.")
                run["client_seen_at"] = now().isoformat()
                return 204, None
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
            return 200, self.evaluation_view(self.evaluations[p[1]])
        raise HTTPError(404, "not_found", f"no route {method} {path}")

    @staticmethod
    def page(items: list, params: dict[str, str]) -> dict:
        limit, offset = int(params.get("limit", 50)), int(params.get("offset", 0))
        return {"items": items[offset : offset + limit], "total": len(items)}

    # ------------------------------------------------------------------ decisions
    def resolve_model(self, model: str) -> tuple[str, str, str | None]:
        """(base model id, what answered, project) for a decision's ``model``, as the API's services/references.py:
        ``<project>`` (its live version, else its base), ``<project>:preview`` (its newest version while that isn't
        live; else 404 no_preview) or
        a live custom model's name."""
        if "/" in model:
            raise HTTPError(
                404,
                "unknown_model",
                f"No model {model!r}: projects are called by their name and custom models by theirs.",
                "model",
            )
        name, sep, env = model.partition(":")
        if sep:
            p = self.projects.get(name)
            if p is None:
                raise HTTPError(404, "unknown_model", f"No project {name!r}.", "model")
            if env == "preview":
                v = self.preview(p)
                if v is None:
                    raise HTTPError(
                        404, "no_preview", f"{name} has no preview: its newest version is live. Call {name}.", "model"
                    )
                return p["base_model"], f"{name}:{v}", name
            if re.fullmatch(r"v[1-9]\d*", env):
                raise HTTPError(
                    404,
                    "version_not_callable",
                    f"{model} isn't callable: call {name} (the live version) or {name}:preview, or promote {env}.",
                    "model",
                )
            raise HTTPError(404, "unknown_model", f"{name} has no environment {env!r}.", "model")
        if model in self.projects:
            p = self.projects[model]
            return p["base_model"], f"{model}:{p['live']}" if p["live"] else model, model  # the base: "<project>"
        if model in BASE_MODELS:
            raise HTTPError(
                422,
                "model_requires_project",
                f"Create a continuous learning project on {model} to decide with it.",
                "model",
            )
        cm = self.models.get(model)
        if cm is None or not cm["_ready"]:
            raise HTTPError(404, "unknown_model", f"No model {model!r}.", "model")
        if cm["live_at"] is None:
            raise HTTPError(409, "model_not_in_production", f"{model} is archived: promote it first.", "model")
        return cm["base_model"], model, None

    def systemone(self, body: dict) -> dict:
        for k in ("model", "state", "questions"):
            if k not in body:
                raise HTTPError(422, "invalid_input", f"missing {k}", k)
        if not 1 <= len(body["questions"]) <= 64:
            raise HTTPError(422, "invalid_input", "send 1..64 questions", "questions")
        base_id, answered, project = self.resolve_model(body["model"])
        base = BASE_MODELS[base_id]
        images = find_images(body["state"])
        if images and "image" not in base.get("modalities", ["text"]):
            raise HTTPError(422, "unsupported_modality", f"{base_id} reads text only", images[0][0])
        for path, im in images:
            if set(im) != {"type", "media_type", "data"} or im["media_type"] not in IMAGE_TYPES:
                raise HTTPError(422, "invalid_input", f"{path}: not an image", path)
        self.check_balance()
        answers = {}
        for name, q in body["questions"].items():
            check_question(q, f"questions.{name}")
            if len(option_keys(q)) > base["max_options"]:
                raise HTTPError(422, "invalid_options", f"questions.{name}: more than {base['max_options']} options")
            answers[name] = answer_from_probs(q, uniform(q))
        self.decisions[self.request_id] = {"project": project, "answers": answers}
        return {
            "id": self.request_id,
            "model": answered,
            "answers": answers,
            "usage": {"input_tokens": (7 + 280 * len(images)) * len(answers), "output_tokens": 0},
        }

    def check_correction(self, item: Any, where: str) -> dict:
        """One correction checked against its decision's answers (as the API's services/corrections.py), not kept:
        an answer must be exactly one of the question's options, in value and type."""
        rid = item.get("request_id") if isinstance(item, dict) else None
        d = self.decisions.get(rid) if isinstance(rid, str) else None
        if d is None:
            raise HTTPError(404, "not_found", f"No decision {rid!r}.", f"{where}request_id" if where else None)
        questions, comment = item.get("questions"), item.get("comment")
        if not isinstance(questions, dict) or not questions:
            raise HTTPError(422, "invalid_input", "send 1 or more questions", f"{where}questions")
        if comment is not None and not (isinstance(comment, str) and len(comment) <= 1000):
            raise HTTPError(422, "invalid_input", "at most 1,000 characters", f"{where}comment")
        out = {}
        for name, q in questions.items():
            a, param = d["answers"].get(name), f"{where}questions.{name}"
            if not isinstance(q, dict) or not isinstance(q.get("correct"), bool):
                raise HTTPError(422, "invalid_input", f"{param}.correct: send true or false", f"{param}.correct")
            kind = a["type"] if a else None
            if kind == "noul":
                options: list = [False, True]
                served: Any = a["noul"] >= 0.5
            elif kind == "choice":
                options, served = list(a["probabilities"]), a["choice"]
            elif kind == "score":
                options = list(range(len(a["probabilities"])))
                served = max(options, key=lambda i: a["probabilities"][str(i)])
            else:
                raise HTTPError(422, "invalid_input", f"the decision has no question {name!r}", param)
            v = q.get("answer")
            if v is not None and not any(v == o and type(v) is type(o) for o in options):
                raise HTTPError(422, "invalid_input", f"{param}.answer: send one of {options}", f"{param}.answer")
            if q["correct"]:
                if v is not None and v != served:
                    raise HTTPError(422, "invalid_input", f"{param}.answer: send correct: false", f"{param}.answer")
                answer = served
            elif v is not None:
                if v == served:
                    raise HTTPError(422, "invalid_input", f"{param}.answer: send correct: true", f"{param}.answer")
                answer = v
            else:
                others = [o for o in options if o != served]
                answer = others[0] if len(others) == 1 else None
            out[name] = {"correct": q["correct"], "answer": answer, "served": served}
        return {
            "request_id": rid,
            "project": d["project"],
            "questions": out,
            "comment": comment,
            "created_at": now().isoformat(),
        }

    def keep_correction(self, c: dict) -> dict:
        """The active correction after `c`: the same questions and comment keep the active one, else `c` replaces it."""
        cur = self.corrections.get(c["request_id"])
        if cur is not None and (cur["questions"], cur["comment"]) == (c["questions"], c["comment"]):
            return cur
        self.corrections[c["request_id"]] = c
        return c

    def list_models(self) -> dict:
        """GET /v1/models: the names a decision can send (`<project>`, `<project>:preview`, live custom models)."""
        items: list[dict] = []
        for name, p in sorted(self.projects.items()):
            refs = [(name, "production")] + ([(f"{name}:preview", "preview")] if self.preview(p) else [])
            for ref, env in refs:
                _, serves, _ = self.resolve_model(ref)
                items.append(
                    {
                        "name": ref,
                        "description": f"{ref}: now {serves}.",
                        "release_date": "2026-10-01",
                        "endor": {
                            "kind": "project",
                            "project": name,
                            "environment": env,
                            "base_model": p["base_model"],
                            "model": serves,
                        },
                    }
                )
        items += [
            {
                "name": cm["name"],
                "description": f"custom model on {cm['base_model']}",
                "release_date": cm["created_at"][:10],
                "endor": {"kind": "custom", "base_model": cm["base_model"], "training_run_id": cm["training_run_id"]},
            }
            for cm in self.models.values()
            if cm["_ready"] and cm["live_at"] is not None
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
                    "modalities": ["text"],
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
        if set(body) - {"name", "description", "base_model"}:  # the SDK sends the contract's fields only
            raise HTTPError(422, "invalid_input", f"unknown fields {sorted(set(body) - {'name', 'description'})}")
        if self.name_taken(name):
            raise HTTPError(409, "conflict", f"{name} is taken by a project, custom model or base model", "name")
        cap = int(LIMITS["max_projects_per_org"])
        if len(self.projects) >= cap:
            raise HTTPError(
                409,
                "limit_reached",
                f"This organization has {len(self.projects)} projects, the maximum of {cap}. Delete one first.",
            )
        base = body.get("base_model")
        if base not in BASE_MODELS:
            raise HTTPError(422, "invalid_input", f"Not a base model: {base!r}.", "base_model")
        p = {
            "name": name,
            "description": body.get("description"),
            "base_model": base,
            "live": None,
            "auto_promote": False,
            "promoted_at": None,
            "created_at": now().isoformat(),
        }
        self.projects[name] = p
        self.versions[name] = []
        return self.project_view(p)

    @staticmethod
    def check_rows(rows: Any) -> tuple[int, Counter]:
        """Decision rows as a dataset upload (or an evaluation's inline rows) checks them: (labelled questions,
        question types)."""
        if not isinstance(rows, list) or not 1 <= len(rows) <= 50_000:
            raise HTTPError(422, "invalid_input", "1..50000 rows", "rows")
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
        return n_labelled, types

    def datasets_route(self, m: str, tail: list[str], body: Any, params: dict) -> tuple[int, Any]:
        if not tail and m == "POST":
            name, rows = body.get("name"), body.get("rows")
            if not isinstance(name, str) or not NAME.match(name):
                raise HTTPError(422, "invalid_input", "bad name", "name")
            if name in self.datasets:
                raise HTTPError(409, "conflict", f"dataset {name} already exists")
            n_labelled, types = self.check_rows(rows)
            d = {
                "name": name,
                "n_rows": len(rows),
                "n_labelled_questions": n_labelled,
                "question_types": dict(types),
                "n_images": self.check_images(rows, "rows."),
                "created_at": now().isoformat(),
                "_rows": rows,
            }
            self.datasets[name] = d
            return 201, {k: v for k, v in d.items() if not k.startswith("_")}
        if not tail and m == "GET":
            items = [{k: v for k, v in d.items() if not k.startswith("_")} for d in reversed(self.datasets.values())]
            return 200, self.page(items, params)
        if tail and tail[0] not in self.datasets:
            raise HTTPError(404, "not_found", f"no dataset {tail[0]}")
        d = self.datasets[tail[0]]
        if len(tail) == 1 and m == "GET":
            return 200, {k: v for k, v in d.items() if not k.startswith("_")}
        if len(tail) == 1 and m == "DELETE":
            del self.datasets[tail[0]]
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

    def create_run(self, body: dict) -> dict:
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
        model = body.get("model")
        if model is not None:
            if not isinstance(model, str) or not NAME.match(model):
                raise HTTPError(422, "invalid_input", "bad model name", "model")
            if self.name_taken(model):
                raise HTTPError(409, "conflict", f"{model} is taken by a project, custom model or base model", "model")
        base_model, parent = body.get("base_model"), None
        if body.get("from_model"):
            pm = self.models.get(body["from_model"])
            if pm is None or not pm["_ready"]:
                raise HTTPError(422, "unknown_model", f"No custom model {body['from_model']!r}.", "from_model")
            if body.get("include_optimizer") and not pm["has_optimizer"]:
                raise HTTPError(409, "invalid_state", f"{pm['name']} was saved without optimizer state")
            if base_model is not None and base_model != pm["base_model"]:
                raise HTTPError(422, "invalid_input", f"{pm['name']} is a {pm['base_model']} model.", "base_model")
            base_model, parent = pm["base_model"], pm["name"]
        if base_model is None:
            raise HTTPError(422, "invalid_input", "Send the base model to train on.", "base_model")
        if base_model not in BASE_MODELS:
            raise HTTPError(422, "invalid_input", f"Not a base model: {base_model!r}.", "base_model")
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
            "name": body.get("name"),
            "model": model,
            "base_model": base_model,
            "contract": "letters-16",
            "lora": lora,
            "status": "provisioning" if self.provisioning_polls else "ready",
            "ready_future_id": ready["id"],
            "step": 0,
            "total_steps": body.get("total_steps"),
            "seconds_per_step": None,
            "ready_at": now().isoformat(),
            "closed_at": None,
            "wandb": body["wandb"] if body.get("wandb") is not None else self.wandb_connected,
            "wandb_url": None,
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
        if self.check_images(data, "data[", base) > MAX_IMAGES_PER_CALL:  # (paths read data[i.state: close enough)
            raise HTTPError(422, "invalid_input", f"at most {MAX_IMAGES_PER_CALL} images in one call", "data")

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

    def predict(self, run: dict, body: dict) -> dict:
        """Decision rows answered as /v1/systemone answers them (uniform), in order; 1..1024 questions per call."""
        rows = body.get("rows")
        if not isinstance(rows, list) or not rows:
            raise HTTPError(422, "invalid_input", "send at least one row", "rows")
        if sum(len(r.get("questions") or {}) for r in rows) > 1024:
            raise HTTPError(422, "invalid_input", "at most 1024 questions per call, across rows", "rows")
        base = BASE_MODELS[run["base_model"]]
        for i, r in enumerate(rows):
            if set(r) - {"state", "questions", "id", "labels", "weight"} or "state" not in r:
                raise HTTPError(422, "invalid_input", "a row is {state, questions}", f"rows.{i}")
            if not isinstance(r.get("questions"), dict) or not 1 <= len(r["questions"]) <= 64:
                raise HTTPError(422, "invalid_input", "send 1..64 questions", f"rows.{i}.questions")
            for name, q in r["questions"].items():
                where = f"rows[{i}].questions.{name}"
                check_question(q, where)
                if len(option_keys(q)) > base["max_options"]:
                    raise HTTPError(422, "invalid_options", f"{where}: more than {base['max_options']} options", where)

        if self.check_images(rows, "rows[", base) > MAX_IMAGES_PER_CALL:
            raise HTTPError(422, "invalid_input", f"at most {MAX_IMAGES_PER_CALL} images in one call", "rows")

        def result() -> dict:
            return {"answers": [{n: answer_from_probs(q, uniform(q)) for n, q in r["questions"].items()} for r in rows]}

        return self.accept(run, body, "predict", result)

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
        if name is None:
            name = run["model"]
            if name is None:
                raise HTTPError(422, "invalid_input", "This run has no model name: send a name.", "name")
        if not isinstance(name, str) or not NAME.match(name):
            raise HTTPError(422, "invalid_input", "bad name", "name")
        ttl = body.get("ttl_seconds")
        if ttl is not None and not 3600 <= ttl <= 10 * 365 * 86400:
            raise HTTPError(422, "invalid_input", "ttl 1h..10y", "ttl_seconds")
        new = "seq_id" in body and body["seq_id"] >= run["next_seq_id"]
        if new and name in BASE_MODELS:  # as the API's models.check_name
            raise HTTPError(422, "invalid_input", f"{name!r} is a base model id.", "name")
        if new and (name in self.projects or name in self.models):
            raise HTTPError(409, "conflict", f"{name} is taken by a project or custom model", "name")

        def result() -> dict:
            self.models[name] = {
                "name": name,
                "training_run_id": run["id"],
                "base_model": run["base_model"],
                "contract": "letters-16",
                "parent_model": run["parent_model"],
                "step": run["step"],
                "has_optimizer": bool(body.get("include_optimizer")),
                "size_bytes": 1024,
                "live_at": None,
                "expires_at": (now() + timedelta(seconds=ttl)).isoformat() if ttl else None,
                "user_metadata": body.get("user_metadata") or {},
                "created_at": now().isoformat(),
                "_ready": True,
            }
            return {"model": name}

        return self.accept(run, body, "save_checkpoint", result)

    # ------------------------------------------------------------------ custom models
    def custom_models_route(self, m: str, tail: list[str], body: Any, params: dict) -> tuple[int, Any]:
        if not tail and m == "GET":
            mine = [x for x in self.models.values() if x["_ready"]]
            if params.get("production") is not None:
                want = params["production"] == "true"
                mine = [x for x in mine if (x["live_at"] is not None) == want]
            live = sorted((x for x in mine if x["live_at"]), key=lambda x: x["live_at"])  # in the order promoted
            rest = sorted((x for x in mine if not x["live_at"]), key=lambda x: x["created_at"], reverse=True)
            return 200, self.page([self.model_view(x) for x in live + rest], params)
        mod = self.custom_model(tail[0])
        if len(tail) == 1 and m == "GET":
            return 200, self.model_view(mod)
        if len(tail) == 1 and m == "PATCH":
            if "ttl_seconds" in body:
                ttl = body["ttl_seconds"]
                if ttl is not None and mod["live_at"] is not None:
                    raise HTTPError(409, "model_in_production", f"{mod['name']} is in production: it never expires.")
                if ttl is not None and not 3600 <= ttl <= 10 * 365 * 86400:
                    raise HTTPError(422, "invalid_input", "ttl 1h..10y", "ttl_seconds")
                mod["expires_at"] = (now() + timedelta(seconds=ttl)).isoformat() if ttl else None
            return 200, self.model_view(mod)
        if len(tail) == 1 and m == "DELETE":
            if mod["live_at"] is not None:
                raise HTTPError(409, "model_in_production", f"{mod['name']} is in production: archive it first.")
            del self.models[mod["name"]]
            return 204, None
        if tail[1:] == ["promote"] and m == "POST":
            if mod["live_at"] is not None:
                if mod.get("_serving") == "failed":  # deployed again
                    mod["_serving"], mod["_serving_error"] = "live", None
                return 200, self.model_view(mod)
            live = [x for x in self.models.values() if x["live_at"] is not None]
            replace = (body or {}).get("replace")
            if replace is not None:
                r = self.models.get(replace)
                if r is None or r["live_at"] is None:
                    raise HTTPError(404, "not_found", f"{replace} isn't a custom model in production.", "replace")
                r["live_at"] = None
                live.remove(r)
            cap = int(LIMITS["max_production_models_per_org"])
            if len(live) >= cap:
                names = ", ".join(x["name"] for x in live)
                raise HTTPError(
                    409, "production_full", f"Every production slot is taken ({cap}): {names}. Send replace with one."
                )
            mod["live_at"], mod["expires_at"] = now().isoformat(), None
            mod["_serving"], mod["_serving_error"] = "live", None  # loaded at once
            return 200, self.model_view(mod)
        if tail[1:] == ["demote"] and m == "POST":
            mod["live_at"] = None
            return 200, self.model_view(mod)
        if tail[1:] == ["download"] and m == "GET":
            return 200, self.download_files(
                mod["name"], mod["has_optimizer"], params.get("include_optimizer") == "true"
            )
        raise HTTPError(404, "not_found", "no route")

    # ------------------------------------------------------------------ evaluations
    def create_evaluation(self, body: dict) -> dict:
        """A dataset's rows or rows sent inline (one of the two), decided at once with uniform distributions, as the
        API: a row with labels is asked (and scored on) its labeled questions, a row without labels every question
        (answered, not scored). ``model`` is a custom model (live or not) or a base model id. rows.jsonl is served
        from a presigned link, its lines in reverse order (the SDK must order them by ``index``)."""
        self.check_balance()
        dataset, rows = body.get("dataset"), body.get("rows")
        if (dataset is None) == (rows is None):
            raise HTTPError(422, "invalid_input", "send dataset or rows", "rows" if dataset else "dataset")
        if dataset is not None:
            if dataset not in self.datasets:
                raise HTTPError(404, "not_found", f"no dataset {dataset}")
            rows = self.datasets[dataset]["_rows"]
        else:
            self.check_rows(rows)
        model = body.get("model")
        if model in BASE_MODELS:
            base = model
        elif model in self.models and self.models[model]["_ready"]:
            base = self.models[model]["base_model"]
        else:
            raise HTTPError(404, "unknown_model", f"No custom model or base model {model!r}.", "model")
        self.check_images(rows, "rows.", BASE_MODELS[base])
        lines, n, by_question = [], 0, Counter()
        for i, row in enumerate(rows):
            labels = row.get("labels") or {}
            asked = {k: q for k, q in row["questions"].items() if k in labels} or row["questions"]
            for name, q in asked.items():
                if len(option_keys(q)) > BASE_MODELS[base]["max_options"]:
                    where = f"rows.{i}.questions.{name}"
                    raise HTTPError(422, "invalid_options", f"{where}: too many options", where)
                probs = uniform(q)
                target = label_target(q, labels[name]) if name in labels else None
                n += target is not None
                by_question[name] += target is not None
                lines.append(
                    {
                        "row": row.get("id"),
                        "index": i,
                        "question": name,
                        "probabilities": dict(zip(option_keys(q), probs, strict=False)),
                        "answer": answer_from_probs(q, probs),
                        "target": target,
                    }
                )
        token = uuid.uuid4().hex
        self.blobs[token] = "".join(json.dumps(x) + "\n" for x in reversed(lines)).encode()
        e = {
            "id": self.new_id("evl"),
            "model": model,
            "dataset": dataset,
            "training_run_id": body.get("training_run_id"),
            "source": "server",
            "status": "completed",
            "results": {
                "n": n,
                "overall": {"n": n, "accuracy": 0.5} if n else {"n": 0},
                "by_question": {q: {"n": k, "accuracy": 0.5} for q, k in by_question.items() if k},
                "rows_url": f"https://{FILES_HOST}/{token}/rows.jsonl?X-Amz-Signature=sig{token}&X-Amz-Expires=3600",
            },
            "rows_total": len(rows),
            "rows_done": len(rows),
            "started_at": now().isoformat(),
            "created_at": now().isoformat(),
        }
        self.evaluations[e["id"]] = e
        f = self.future("evaluate", {"evaluation_id": e["id"]})
        e["_future"] = f["id"]
        return {"future_id": f["id"], "evaluation_id": e["id"]}

    def evaluation_view(self, e: dict) -> dict:
        """An evaluation as the API shows it: halfway (running, no results) while its future is open."""
        out = {k: v for k, v in e.items() if not k.startswith("_")}
        if "_future" in e and self.futures[e["_future"]]["status"] in ("pending", "running"):
            out.update(status="running", rows_done=e["rows_total"] // 2, results=None)
        return out
