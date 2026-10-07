from __future__ import annotations

import pytest

import endor
from endor import APIFuture, OperationFailedError

from .conftest import rows
from .fake_api import FakeEndor


def datums(n: int = 2) -> list[endor.Datum]:
    return endor.data.rows_to_datums(rows(n))


def test_long_poll_until_completed(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 3
    run = project.runs.create("jevk5-4b")
    fut = run.forward(datums())
    assert not fut.done() and fut.info is None
    out = fut.result()
    polls = [r for r in fake.requests if r.path == f"/v1/futures/{fut.id}"]
    assert len(polls) == 4
    assert all(0 < float(r.params["wait_s"]) <= 25 for r in polls)
    assert out.metrics["n"] == 4 and fut.done() and fut.info is not None and fut.info.status == "completed"
    assert fut.result() is out  # cached
    run.close()


def test_timeout_leaves_the_operation_running(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 10**6
    run = project.runs.create("jevk5-4b")
    fut = run.forward(datums())
    with pytest.raises(TimeoutError, match=str(fut.id)):
        fut.result(timeout=0.01)
    assert not fut.done()
    fake.futures[fut.id or ""]["_polls"] = 0
    assert fut.result(timeout=1).metrics["n"] == 4
    run.close()


def test_cancel(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 5
    run = project.runs.create("jevk5-4b")
    fut = run.forward(datums())
    fut.cancel()
    assert fake.requests[-1].path == f"/v1/futures/{fut.id}/cancel"
    with pytest.raises(OperationFailedError) as e:
        fut.result()
    assert e.value.code == "cancelled"
    n = len(fake.requests)
    done = APIFuture.completed(1)
    done.cancel()  # no request for a settled future
    fut.cancel()
    assert len(fake.requests) == n
    run.close()


def test_gather_uses_one_retrieve_per_round(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 2
    run = project.runs.create("jevk5-4b")
    fb = run.forward_backward(datums(3))
    opt = run.optim_step(learning_rate=1e-3)
    out, step = endor.gather(fb, opt)
    assert out.metrics["n"] == 6 and step.step == 1 and step.learning_rate == 1e-3
    retrieves = [r for r in fake.requests if r.path == "/v1/futures/retrieve"]
    assert len(retrieves) == 3 and all(len(r.body["ids"]) == 2 for r in retrieves)
    assert endor.gather() == []
    assert endor.gather(APIFuture.completed("x"), fb) == ["x", out]
    run.close()


def test_gather_reports_the_first_failure_after_all_settle(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 1
    run = project.runs.create("jevk5-4b")
    a = run.forward_backward(datums())
    b = run.optim_step()
    fake.fail_future(a.id or "", "oom", "out of memory")
    with pytest.raises(OperationFailedError) as e:
        endor.gather(a, b)
    assert e.value.code == "oom" and b.done()
    run.close()


async def test_await_and_gather_async(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 2
    run = project.runs.create("jevk5-4b")
    fb = await run.forward_backward_async(datums())
    opt = await run.optim_step_async()
    out = await fb
    assert out.metrics["n"] == 4
    (step,) = await endor.gather_async(opt)
    assert step.step == 1
    fut = await run.forward_async(datums())
    await fut.cancel_async()
    with pytest.raises(OperationFailedError):
        await fut.result_async(timeout=1)
    await run.close_async()


def test_completed_future() -> None:
    f: APIFuture[int] = APIFuture.completed(42)
    assert f.done() and f.result() == 42 and f.leaves() == [] and "done=True" in repr(f)


def test_missing_future_is_not_found(client: endor.EndorClient) -> None:
    f: APIFuture[dict] = APIFuture(client._t, "fut_nope")
    with pytest.raises(endor.NotFoundError):
        f.result()


def test_many_leaves_are_chunked(project: endor.Project, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endor.futures, "MAX_FUTURES_PER_RETRIEVE", 2)
    run = project.runs.create("jevk5-4b")
    futs = [run.forward(datums(1)) for _ in range(5)]
    outs = endor.gather(*futs)
    assert [o.metrics["n"] for o in outs] == [2] * 5
    retrieves = [r for r in fake.requests if r.path == "/v1/futures/retrieve"]
    assert [len(r.body["ids"]) for r in retrieves] == [2, 2, 1]
    run.close()
