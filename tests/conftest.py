from __future__ import annotations

import itertools
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

_counter = itertools.count()


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
    return f"{prefix}-{next(_counter)}"


@pytest.fixture
def fake() -> FakeEndor:
    return FakeEndor()


@pytest.fixture
def client(fake: FakeEndor) -> Iterator[endor.EndorClient]:
    with endor.EndorClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        transport=httpx.MockTransport(fake.handler),
        async_transport=httpx.MockTransport(fake.handler),
        retry=endor.RetryPolicy(backoff_initial=0.0, backoff_max=0.0),
    ) as c:
        yield c


@pytest.fixture
def project(client: endor.EndorClient) -> endor.Project:
    return client.projects.create(unique("tickets"))


def requests_to(fake: FakeEndor, path_suffix: str, method: str | None = None) -> list:
    return [r for r in fake.requests if r.path.endswith(path_suffix) and (method is None or r.method == method)]
