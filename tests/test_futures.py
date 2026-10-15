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
    run = project.runs.create("jev-9b")
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
    run = project.runs.create("jev-9b")
    fut = run.forward(datums())
    with pytest.raises(TimeoutError, match=str(fut.id)):
        fut.result(timeout=0.01)
    assert not fut.done()
    fake.futures[fut.id or ""]["_polls"] = 0
    assert fut.result(timeout=1).metrics["n"] == 4
    run.close()


def test_cancel(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 5
    run = project.runs.create("jev-9b")
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
    run = project.runs.create("jev-9b")
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
    run = project.runs.create("jev-9b")
    a = run.forward_backward(datums())
    b = run.optim_step()
    fake.fail_future(a.id or "", "oom", "out of memory")
    with pytest.raises(OperationFailedError) as e:
        endor.gather(a, b)
    assert e.value.code == "oom" and b.done()
    run.close()


async def test_await_and_gather_async(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 2
    run = project.runs.create("jev-9b")
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
    run = project.runs.create("jev-9b")
    futs = [run.forward(datums(1)) for _ in range(5)]
    outs = endor.gather(*futs)
    assert [o.metrics["n"] for o in outs] == [2] * 5
    retrieves = [r for r in fake.requests if r.path == "/v1/futures/retrieve"]
    assert [len(r.body["ids"]) for r in retrieves] == [2, 2, 1]
    run.close()


def one_step(run: endor.Run) -> tuple[APIFuture[endor.ForwardOutput], APIFuture[endor.OptimStepOutput]]:
    return run.forward_backward(datums()), run.optim_step(learning_rate=1e-3)


def step_calls(fake: FakeEndor) -> list[str]:
    """The training calls and future polls so far, by the last part of their path."""
    return [
        r.path.rsplit("/", 1)[-1]
        for r in fake.requests
        if r.path.endswith(("/forward_backward", "/optim_step")) or r.path.startswith("/v1/futures")
    ]


def test_steps_send_the_next_step_before_reading_the_last(project: endor.Project, fake: FakeEndor) -> None:
    with project.runs.create("jev-9b") as run:
        calls = len(step_calls(fake))  # the run's ready future
        steps = endor.Steps()  # 2 in flight
        assert steps.add(*one_step(run)) == [] and len(steps) == 1
        done = steps.add(*one_step(run))  # step 2 is sent, then step 1 is read
        assert [(out.metrics["n"], opt.step) for out, opt in done] == [(4, 1)] and len(steps) == 1
        assert step_calls(fake)[calls:] == [
            "forward_backward",
            "optim_step",
            "forward_backward",
            "optim_step",
            "retrieve",
        ]
        assert [opt.step for _, opt in steps.wait()] == [2] and len(steps) == 0
        assert steps.wait() == []


def test_steps_with_one_in_flight_wait_for_each_step(project: endor.Project, fake: FakeEndor) -> None:
    with project.runs.create("jev-9b") as run:
        calls = len(step_calls(fake))
        steps = endor.Steps(in_flight=1)
        assert [opt.step for _, opt in steps.add(*one_step(run))] == [1]
        assert [opt.step for _, opt in steps.add(*one_step(run))] == [2]
        assert steps.wait() == []
        assert step_calls(fake)[calls:] == ["forward_backward", "optim_step", "retrieve"] * 2
        with pytest.raises(ValueError, match="in_flight"):
            endor.Steps(in_flight=0)


def test_a_failed_step_raises_where_it_is_read(project: endor.Project, fake: FakeEndor) -> None:
    with project.runs.create("jev-9b") as run:
        steps = endor.Steps()
        steps.add(*one_step(run))
        fb, opt = one_step(run)
        fake.fail_future(fb.id or "", "oom", "out of memory")
        assert [o.step for _, o in steps.add(fb, opt)] == [1]  # step 1 is read; the failed step 2 is still in flight
        with pytest.raises(OperationFailedError) as e:
            steps.add(*one_step(run))  # step 3 is sent, then step 2 is read
        assert e.value.code == "oom" and len(steps) == 1
        assert [o.step for _, o in steps.wait()] == [3]


async def test_steps_async(project: endor.Project, fake: FakeEndor) -> None:
    fake.future_polls = 1
    async with project.runs.create("jev-9b") as run:
        steps = endor.Steps()
        read = []
        for _ in range(3):
            fb = await run.forward_backward_async(datums())
            opt = await run.optim_step_async(learning_rate=1e-3)
            read += await steps.add_async(fb, opt)
            assert len(steps) == 1
        read += await steps.wait_async()
        assert [opt.step for _, opt in read] == [1, 2, 3] and len(steps) == 0
