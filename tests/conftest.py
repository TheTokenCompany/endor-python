"""Test setup. Every test runs against the in-memory fake (tests/fake_api.py): many check what the fake saw or make
it fail, and they create fixed names, so the suite isn't meant for a real API. tests/test_integration.py is: it runs
against a real API when ENDOR_TEST_BASE_URL and ENDOR_TEST_API_KEY are set, and is skipped otherwise.

Every test must close every run it opens: the fake, like the API, refuses runs over the open-run limit (the fake's
is 4; the API's is 25 by default, and ``client.whoami().limits["max_active_runs"]`` has an org's own), and the suite
fails a test that leaves one open.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

import httpx
import pytest

import endor
from endor import Choice, Noul, Score

from .fake_api import FakeEndor

API_KEY = "edk_test"
BASE_URL = "https://endor.test"

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
    """A project or custom model name no other test or earlier run of the suite uses."""
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
def fake() -> Iterator[FakeEndor]:
    """The fake API itself, for tests that inspect requests or inject faults. It fails a test that leaves a run
    open."""
    f = FakeEndor()
    yield f
    still_open = [r["id"] for r in f.runs.values() if r["status"] not in DONE]
    assert not still_open, f"the test left runs open: {still_open}"


@pytest.fixture
def client(fake: FakeEndor) -> Iterator[endor.EndorClient]:
    c = endor.EndorClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        transport=httpx.MockTransport(fake.handler),
        async_transport=httpx.MockTransport(fake.handler),
        retry=endor.RetryPolicy(backoff_initial=0.0, backoff_max=0.0),
    )
    with c:
        yield c


BASE = "jev-9b"
"""The base model the tests train custom models on."""


@pytest.fixture
def project(client: endor.EndorClient) -> endor.Project:
    """A new continuous learning project on jev-9b."""
    return client.projects.create(unique("tickets"), base_model=BASE)


DECIDE_BASE = BASE


@pytest.fixture
def decider(client: endor.EndorClient) -> str:
    """The name of a new continuous learning project: ``model=decider`` answers with its base model (no version
    yet)."""
    return client.projects.create(unique("decide"), base_model=DECIDE_BASE).name


def requests_to(fake: FakeEndor, path_suffix: str, method: str | None = None) -> list:
    return [r for r in fake.requests if r.path.endswith(path_suffix) and (method is None or r.method == method)]
