"""Test setup. By default every test runs against the in-memory fake (tests/fake_api.py).

Set ENDOR_TEST_BASE_URL and ENDOR_TEST_API_KEY to run the same suite against a real API instead: the ``client`` and
``project`` fixtures then talk to that server, and tests that need the fake itself (they inspect its requests or
inject faults) are skipped. Either way, every test must close every run it opens: the API allows 4 open runs per org.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import endor
from endor import Choice, Noul, Score

from .fake_api import FakeEndor

API_KEY = "edk_test"
BASE_URL = "https://endor.test"
LIVE_URL = os.environ.get("ENDOR_TEST_BASE_URL")
LIVE_KEY = os.environ.get("ENDOR_TEST_API_KEY")
LIVE = bool(LIVE_URL and LIVE_KEY)

DEPT = Choice(instructions="Route the ticket", criteria={"billing": "money", "tech": None})
URGENT = Noul(instructions="Is it urgent?")
ANGER = Score(instructions="How angry?", criteria=["calm", "annoyed", "furious"])

# Statuses of a run that no longer holds (or will soon release) its slot.
DONE = ("closed", "failed")


def rows(n: int = 40) -> list[dict]:
    """Labeled native rows with a choice and a noul question each."""
    return [
        {
            "id": f"r{i}",
            "state": {"body": f"ticket {i}"},
            "questions": {"dept": DEPT, "urgent": URGENT},
            "labels": {"dept": "billing" if i % 2 else "tech", "urgent": bool(i % 3)},
        }
        for i in range(n)
    ]


def unique(prefix: str = "p") -> str:
    """A project name no other test or earlier run of the suite uses."""
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def wait_closed(run: endor.Run, timeout: float = 10.0) -> str:
    """The run's status once it has finished closing (``closing`` lasts until accepted calls finish)."""
    deadline = time.monotonic() + timeout
    while True:
        status = run.info().status
        if status in DONE or time.monotonic() > deadline:
            return status
        time.sleep(0.1)


@pytest.fixture
def _fake() -> Iterator[FakeEndor]:
    f = FakeEndor()
    yield f
    still_open = [r["id"] for r in f.runs.values() if r["status"] not in DONE]
    assert not still_open, f"the test left runs open: {still_open}"


@pytest.fixture
def fake(_fake: FakeEndor) -> FakeEndor:
    """The fake API itself, for tests that inspect requests or inject faults (skipped against a real API)."""
    if LIVE:
        pytest.skip("needs the in-memory fake")
    return _fake


@pytest.fixture
def client(_fake: FakeEndor) -> Iterator[endor.EndorClient]:
    retry = endor.RetryPolicy(backoff_initial=0.0, backoff_max=0.0)
    if LIVE:
        assert LIVE_URL and LIVE_KEY
        c = endor.EndorClient(api_key=LIVE_KEY, base_url=LIVE_URL, retry=retry)
    else:
        c = endor.EndorClient(
            api_key=API_KEY,
            base_url=BASE_URL,
            transport=httpx.MockTransport(_fake.handler),
            async_transport=httpx.MockTransport(_fake.handler),
            retry=retry,
        )
    started = datetime.now(timezone.utc) - timedelta(seconds=5)
    with c:
        yield c
        if LIVE:
            _assert_no_open_runs(c, started)


def _assert_no_open_runs(c: endor.EndorClient, since: datetime) -> None:
    """Runs left open in the projects this test created (the projects list is newest first)."""
    projects = [p for p in c.projects.list(limit=50) if p.info_.created_at >= since]
    deadline = time.monotonic() + 10
    while True:
        still_open = [r.id for p in projects for r in p.runs.list() if r.info_.status not in DONE]
        if not still_open or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    assert not still_open, f"the test left runs open: {still_open}"


@pytest.fixture
def project(client: endor.EndorClient) -> endor.Project:
    """A new project without a base model (its first run sets one)."""
    return client.projects.create(unique("tickets"))


DECIDE_BASE = "jev-9b"


@pytest.fixture
def decider(client: endor.EndorClient) -> str:
    """The name of a new project with a base model: ``model=decider`` answers with its live model (the base)."""
    return client.projects.create(unique("decide"), base_model=DECIDE_BASE).name


def requests_to(fake: FakeEndor, path_suffix: str, method: str | None = None) -> list:
    return [r for r in fake.requests if r.path.endswith(path_suffix) and (method is None or r.method == method)]
