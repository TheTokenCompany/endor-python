"""``client.base_models()`` and ``endor base-models`` on ``GET /v1/base_models`` as the API sends it."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from endor import DecisionIndexInfo, EndorClient, cli

from .conftest import API_KEY, BASE_URL

INDEX_URL = "https://huggingface.co/spaces/multimodalart/jev-decision-index"

# Entries of GET /v1/base_models as the API sends them (October 2026), task_ranks cut to three benchmarks.
PPLX: dict[str, Any] = {
    "id": "perplexity-ai/pplx-decider-v1.1-27b",
    "params": "27B",
    "provider": "Perplexity",
    "leaderboard_rank": 1,
    "decision_index": {
        "name": "Perplexity Decider v1.1 (27B)",
        "source_url": "https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b",
        "rank": 1,
        "tied": False,
        "meant_for": ["Code", "Agents", "Documents and email", "Finance"],
        "use_case_ranks": {"Code": 1, "Finance": 14.2, "Documents and email": 12.6, "Agents": 6.4},
        "task_ranks": {"function-call decisions": 1, "phishing detection": 16, "reading contracts": 17},
        "note": None,
        "engine": "pplx-decider-v1.1-27b",
        "url": f"{INDEX_URL}?model=pplx-decider-v1.1-27b",
        "edition": "Decision Index 0.3.1",
        "updated": "2026-10-10",
        "entrants": 117,
    },
    "tokens_per_s": 5921,
    "tokens_per_s_source": "Measured on real traffic",
    "hf_repo": "perplexity-ai/pplx-decider-v1.1-27b",
    "hf_revision": "3b45dead91dfa6d95aad6b95764a606fab2bf7a6",
    "contract": "pplx-readout-255",
    "contract_version": 2,
    "max_options": 255,
    "trainable": True,
    "lora_targets": ["attn", "mlp", "deltanet"],
    "default_rank": 16,
    "max_rank": 64,
    "modalities": ["text", "image"],
    "max_question_tokens": 16384,
    "price_per_mtok_decide": 0.02,
    "price_per_mtok_decide_continuous_learning": 0.03,
    "price_per_gpu_hour": 7.0,
}
GEV: dict[str, Any] = {
    "id": "autotrust/gev-26b",
    "params": "26B (4B active)",
    "provider": "AutoTrust",
    "leaderboard_rank": 14,
    "decision_index": {
        "name": "GEV-26B-Decide",
        "source_url": "https://huggingface.co/autotrust/GEV-26B-Decide",
        "rank": 14,
        "tied": True,
        "meant_for": ["Code", "Documents and email", "Finance", "Agents"],
        "use_case_ranks": {"Code": 13, "Finance": 17.2, "Documents and email": 16.2, "Agents": 28},
        "task_ranks": {"function-call decisions": 38, "phishing detection": 13, "reading contracts": 5},
        "note": None,
        "engine": "jev-gemma4-26b-a4b",
        "url": f"{INDEX_URL}?model=jev-gemma4-26b-a4b",
        "edition": "Decision Index 0.3.1",
        "updated": "2026-10-10",
        "entrants": 117,
    },
    "tokens_per_s": 13791,
    "tokens_per_s_source": "Measured on real traffic",
    "hf_repo": "autotrust/GEV-26B-Decide",
    "hf_revision": "7c89590ead085bf77630b4bf68264ea30b6ddc78",
    "contract": "autotrust-bare-v1",
    "contract_version": 1,
    "max_options": 16,
    "trainable": True,
    "lora_targets": ["attn", "mlp"],
    "default_rank": 16,
    "max_rank": 64,
    "modalities": ["text", "image"],
    "max_question_tokens": 16384,
    "price_per_mtok_decide": 0.042,
    "price_per_mtok_decide_continuous_learning": 0.063,
    "price_per_gpu_hour": 7.0,
}
DECIDER_2B: dict[str, Any] = {
    "id": "mapika/decider-2b",
    "params": "1.9B",
    "provider": "Mapika",
    "leaderboard_rank": 81,
    "decision_index": {
        "name": "Decider 2B",
        "source_url": "https://huggingface.co/Mapika/decider-2b",
        "rank": 81,
        "tied": True,
        "meant_for": [],
        "use_case_ranks": {"Code": 74, "Finance": 79.2, "Documents and email": 73.1, "Agents": 76.2},
        "task_ranks": {"function-call decisions": 78, "phishing detection": 85, "reading contracts": 70},
        "note": "The index ran it in FP8; Endor serves the bf16 weights.",
        "engine": "decider-2b-fp8-http",
        "url": f"{INDEX_URL}?model=decider-2b-fp8-http",
        "edition": "Decision Index 0.3.1",
        "updated": "2026-10-10",
        "entrants": 117,
    },
    "tokens_per_s": 45417,
    "tokens_per_s_source": "Measured on real traffic",
    "hf_repo": "Mapika/decider-2b",
    "hf_revision": "533964dae8be954c5b5e19fa4948e48408094c1e",
    "contract": "decider-letters-255",
    "contract_version": 1,
    "max_options": 255,
    "trainable": True,
    "lora_targets": ["attn", "mlp", "deltanet"],
    "default_rank": 16,
    "max_rank": 256,
    "modalities": ["text"],
    "max_question_tokens": 16384,
    "price_per_mtok_decide": 0.007,
    "price_per_mtok_decide_continuous_learning": 0.0105,
    "price_per_gpu_hour": 7.0,
}
# A base that isn't on the index and wasn't measured: the API sends null.
OFF_INDEX: dict[str, Any] = {
    "id": "autotrust/jev-9b",
    "params": "9B",
    "provider": "AutoTrust",
    "leaderboard_rank": None,
    "decision_index": None,
    "tokens_per_s": None,
    "tokens_per_s_source": None,
    "hf_repo": "autotrust/JEV-9B",
    "hf_revision": "b63f651ce8ed64481d3f5e73ecdb05f740042f01",
    "contract": "autotrust-bare-v1",
    "contract_version": 1,
    "max_options": 16,
    "trainable": True,
    "lora_targets": ["attn", "mlp", "deltanet"],
    "default_rank": 16,
    "max_rank": 256,
    "modalities": ["text"],
    "max_question_tokens": 16384,
    "price_per_mtok_decide": 0.02,
    "price_per_mtok_decide_continuous_learning": 0.03,
    "price_per_gpu_hour": 7.0,
}
# An API from before decision_index: the new fields are missing, and it still sent trainer_gpu.
NEW_FIELDS = ("leaderboard_rank", "decision_index", "tokens_per_s", "tokens_per_s_source")
OLDER: dict[str, Any] = {
    **{k: v for k, v in OFF_INDEX.items() if k not in NEW_FIELDS},
    "id": "alibiserikbay/jevk5-9b",
    "trainer_gpu": "H100",
}

TABLE = """\
id                                      rank  tokens/s  context  $/1M tokens  meant for
perplexity-ai/pplx-decider-v1.1-27b    1/117     5,921   16,384        $0.02  Code, Agents, Documents and email, Finance
autotrust/gev-26b                    =14/117    13,791   16,384       $0.042  Code, Documents and email, Finance, Agents
mapika/decider-2b                    =81/117    45,417   16,384       $0.007  -
autotrust/jev-9b                           -         -   16,384        $0.02  -
"""


def client_for(*bases: dict[str, Any]) -> EndorClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/base_models"
        return httpx.Response(200, json={"base_models": list(bases)})

    return EndorClient(api_key=API_KEY, base_url=BASE_URL, transport=httpx.MockTransport(handler))


def run_cli(capsys: pytest.CaptureFixture[str], bases: list[dict[str, Any]], *args: str) -> str:
    with client_for(*bases) as client:
        assert cli.main([*args, "base-models"], client=client) == 0
    return capsys.readouterr().out


def test_decision_index_and_speed() -> None:
    with client_for(PPLX, GEV, DECIDER_2B) as client:
        pplx, gev, small = client.base_models()
    i = pplx.decision_index
    assert isinstance(i, DecisionIndexInfo)
    assert (i.rank, i.entrants, i.tied, pplx.leaderboard_rank) == (1, 117, False, 1)
    assert i.meant_for == ["Code", "Agents", "Documents and email", "Finance"]
    assert i.use_case_ranks == {"Code": 1.0, "Finance": 14.2, "Documents and email": 12.6, "Agents": 6.4}
    assert i.task_ranks["phishing detection"] == 16
    assert i.url == f"{INDEX_URL}?model=pplx-decider-v1.1-27b" and i.engine == "pplx-decider-v1.1-27b"
    assert i.name == "Perplexity Decider v1.1 (27B)" and i.source_url and i.note is None
    assert (i.edition, i.updated) == ("Decision Index 0.3.1", "2026-10-10")
    assert i.model_dump() == PPLX["decision_index"]  # every field the API sends
    assert (pplx.tokens_per_s, pplx.tokens_per_s_source) == (5921, "Measured on real traffic")
    assert gev.decision_index and gev.decision_index.tied
    assert gev.decision_index.meant_for == ["Code", "Documents and email", "Finance", "Agents"]
    assert small.decision_index and small.decision_index.meant_for == []
    assert small.decision_index.note == "The index ran it in FP8; Endor serves the bf16 weights."
    assert small.tokens_per_s == 45417


def test_without_decision_index() -> None:
    with client_for(OFF_INDEX, OLDER) as client:
        bases = client.base_models()
    assert [b.id for b in bases] == ["autotrust/jev-9b", "alibiserikbay/jevk5-9b"]
    for b in bases:
        assert b.decision_index is None and b.leaderboard_rank is None
        assert b.tokens_per_s is None and b.tokens_per_s_source is None
        assert b.max_options == 16 and b.max_question_tokens == 16384 and b.price_per_mtok_decide == 0.02
        assert not hasattr(b, "trainer_gpu") and "trainer_gpu" not in b.model_dump()


def test_decision_index_tolerates_fields_added_or_left_out() -> None:
    i = DecisionIndexInfo.model_validate({"rank": 3, "entrants": 117, "score": 0.91})
    assert (i.rank, i.entrants, i.tied, i.meant_for, i.use_case_ranks, i.task_ranks) == (3, 117, False, [], {}, {})
    assert i.note is None and i.url is None and not hasattr(i, "score")


def test_cli_table(capsys: pytest.CaptureFixture[str]) -> None:
    out = run_cli(capsys, [PPLX, GEV, DECIDER_2B, OFF_INDEX])
    assert out == TABLE
    assert max(len(line) for line in out.splitlines()) <= 120


def test_cli_json_has_every_field(capsys: pytest.CaptureFixture[str]) -> None:
    sent = [PPLX, GEV, DECIDER_2B, OFF_INDEX]
    got = json.loads(run_cli(capsys, sent, "-f", "json"))
    assert [{k: g[k] for k in s} for g, s in zip(got, sent, strict=True)] == sent
    assert all("trainer_gpu" not in g for g in got)


def test_cli_no_base_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(capsys, []) == "No base models\n"
