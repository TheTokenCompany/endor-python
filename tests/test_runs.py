"""Runs: sequence numbers, chunking, lifecycle and dashboard calls."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

import endor
from endor import AdamParams, ConflictError, Datum, OperationFailedError, Target, UnprocessableEntityError

from .conftest import DEPT, rows
from .fake_api import FakeEndor


def datums(n: int = 2) -> list[Datum]:
    return endor.data.rows_to_datums(rows(n))


def seq_ids(fake: FakeEndor, op: str) -> list[int]:
    return [r.body["seq_id"] for r in fake.requests if r.method == "POST" and r.path.endswith(f"/{op}")]


class TestSequencing:
    def test_consecutive_sequence_numbers(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jevk5-4b")
        assert run.next_seq_id == 0
        run.forward_backward(datums()).result()
        run.optim_step().result()
        run.forward(datums()).result()
        run.save_checkpoint("v1").result()
        assert run.next_seq_id == 4
        posts = [r.body["seq_id"] for r in fake.requests if r.method == "POST" and "seq_id" in (r.body or {})]
        assert posts == [0, 1, 2, 3]
        assert run.info().next_seq_id == 4

    def test_second_handle_continues_the_sequence(self, project: endor.Project) -> None:
        run = project.runs.create("jevk5-4b")
        run.forward_backward(datums()).result()
        again = project.runs.get(run.id)
        assert again.next_seq_id == 1
        again.optim_step().result()
        assert again.next_seq_id == 2

    def test_two_handles_conflict(self, project: endor.Project) -> None:
        a = project.runs.create("jevk5-4b")
        b = project.runs.get(a.id)
        a.optim_step(learning_rate=1e-4).result()
        with pytest.raises(ConflictError) as e:
            b.optim_step(learning_rate=2e-4)
        assert e.value.code == "seq_conflict" and "expected seq_id 1" in str(e.value)
        assert b.next_seq_id == 0  # a rejected call does not advance the handle

    def test_transport_retry_reuses_seq_id_and_body(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jevk5-4b")
        fake.fail_next.append(httpx.ConnectError("reset"))
        out = run.forward_backward(datums(3)).result()
        posts = [r for r in fake.requests if r.path.endswith("/forward_backward")]
        assert len(posts) == 2 and posts[0].body == posts[1].body and posts[1].body["seq_id"] == 0
        assert out.metrics["n"] == 6 and run.next_seq_id == 1

    def test_server_side_idempotent_retry(self, project: endor.Project, fake: FakeEndor) -> None:
        """The first POST reached the server (accepted) but the response was lost: the retry gets the same future."""
        run = project.runs.create("jevk5-4b")
        fake.fail_next.append(httpx.ReadTimeout("lost response"))
        # The fault fires before the fake processes the request, so emulate acceptance by replaying by hand:
        run.optim_step().result()
        assert run.info().next_seq_id == 1
        # Now a literal duplicate of an accepted body is answered with the existing future, not a 409.
        body = {"adam_params": AdamParams().model_dump(), "seq_id": 0}
        ref = run._t.request("POST", f"/v1/runs/{run.id}/optim_step", json=body)
        assert ref["future_id"] == fake.ops[(run.id, 0)][1]

    def test_rejected_call_does_not_consume_a_number(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jevk5-4b")
        bad = Datum(state="x", question=DEPT, target=Target(label="nope"))
        with pytest.raises(UnprocessableEntityError) as e:
            run.forward_backward([bad])
        assert e.value.code == "invalid_datum" and "data[0]" in str(e.value)
        assert run.next_seq_id == 0
        run.forward_backward(datums()).result()
        assert seq_ids(fake, "forward_backward") == [0, 0]

    def test_closed_run_rejects_ops(self, project: endor.Project) -> None:
        run = project.runs.create("jevk5-4b")
        run.close()
        with pytest.raises(ConflictError) as e:
            run.optim_step()
        assert e.value.code == "invalid_state"

    async def test_async_calls_are_serialized(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jevk5-4b")
        f1, f2, f3 = await asyncio.gather(
            run.forward_backward_async(datums(2)),
            run.forward_backward_async(datums(3)),
            run.optim_step_async(learning_rate=1e-4),
        )
        outs = await endor.gather_async(f1, f2, f3)
        assert outs[0].metrics["n"] == 4 and outs[1].metrics["n"] == 6 and outs[2].step == 1
        assert sorted(seq_ids(fake, "forward_backward") + seq_ids(fake, "optim_step")) == [0, 1, 2]
        model = await (await run.save_checkpoint_async("v1")).result_async()
        assert model == f"{project.name}/v1"
        assert (await run.close_async()).status == "closed"


class TestChunking:
    def test_big_batches_are_split_and_merged(
        self, project: endor.Project, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(endor.runs, "MAX_DATUMS_PER_CALL", 7)
        run = project.runs.create("jevk5-4b")
        fut = run.forward_backward(datums(10))  # 20 datums -> 7 + 7 + 6
        assert fut.id is None and len(fut.leaves()) == 3
        out = fut.result()
        assert out.metrics["n"] == 20 and len(out.outputs) == 20 and len(out.probabilities) == 20
        assert out.metrics["weight:sum"] == 20 and abs(out.loss - 0.6931) < 1e-3
        assert seq_ids(fake, "forward_backward") == [0, 1, 2]
        assert [len(r.body["data"]) for r in fake.requests if r.path.endswith("/forward_backward")] == [7, 7, 6]
        retrieves = [r for r in fake.requests if r.path == "/v1/futures/retrieve"]
        assert len(retrieves) == 1 and len(retrieves[0].body["ids"]) == 3
        assert fut.done()

    def test_merge_outputs_math(self) -> None:
        a = endor.ForwardOutput(
            outputs=[{"probabilities": {}, "loss": 1.0}],
            metrics={"loss:sum": 2.0, "weight:sum": 2.0, "loss:mean": 1.0, "accuracy": 1.0, "n": 1},
        )
        b = endor.ForwardOutput(
            outputs=[{"probabilities": {}, "loss": 4.0}] * 3,
            metrics={"loss:sum": 6.0, "weight:sum": 1.0, "loss:mean": 6.0, "accuracy": 0.0, "n": 3},
        )
        m = endor.runs.merge_outputs([a, b])
        assert m.metrics == {"loss:sum": 8.0, "weight:sum": 3.0, "loss:mean": 8 / 3, "accuracy": 0.25, "n": 4}
        assert m.losses == [1.0, 4.0, 4.0, 4.0]

    def test_empty_batch(self, project: endor.Project) -> None:
        run = project.runs.create("jevk5-4b")
        with pytest.raises(ValueError, match="no data"):
            run.forward([])


class TestLifecycle:
    def test_with_block_closes(self, project: endor.Project) -> None:
        with project.runs.create("jevk5-4b") as run:
            run.optim_step()
        assert run.info().status == "closed"

    def test_close_warns_about_unsaved_steps(self, project: endor.Project, caplog: pytest.LogCaptureFixture) -> None:
        run = project.runs.create("jevk5-4b")
        run.forward_backward(datums()).result()
        run.optim_step().result()
        with caplog.at_level(logging.WARNING, logger="endor"):
            run.close()
        assert any("newer than the last save" in r.getMessage() for r in caplog.records)

    def test_close_is_quiet_after_a_save(self, project: endor.Project, caplog: pytest.LogCaptureFixture) -> None:
        run = project.runs.create("jevk5-4b")
        run.optim_step().result()
        run.save_checkpoint("v1").result()
        with caplog.at_level(logging.WARNING, logger="endor"):
            run.close()
        assert not caplog.records
        fresh = project.runs.create("jevk5-4b")
        with caplog.at_level(logging.WARNING, logger="endor"):
            fresh.close()
        assert not caplog.records

    def test_failed_op(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jevk5-4b")
        fake.future_polls = 1
        fut = run.forward_backward(datums())
        fake.fail_future(fut.id or "", "trainer_lost", "the trainer died")
        with pytest.raises(OperationFailedError) as e:
            fut.result()
        assert e.value.code == "trainer_lost" and "from_model" in str(e.value) and not e.value.retryable
        assert e.value.future and e.value.future["status"] == "failed"
        with pytest.raises(OperationFailedError):
            fut.result()  # the failure is remembered

    def test_oom_is_retryable(self) -> None:
        e = OperationFailedError("x", code="oom")
        assert e.retryable and "smaller batch" in str(e)

    def test_save_conflict_and_options(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jevk5-4b")
        run.save_checkpoint("v1", include_optimizer=True, ttl_seconds=7200, user_metadata={"note": "x"}).result()
        m = project.models.get("v1")
        assert m.has_optimizer and m.expires_at is not None and m.user_metadata == {"note": "x"}
        with pytest.raises(ConflictError) as e:
            run.save_checkpoint("v1")
        assert e.value.code == "conflict"
        with pytest.raises(UnprocessableEntityError):
            run.save_checkpoint("v2", ttl_seconds=10)

    def test_optim_step_params(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jevk5-4b")
        out = run.optim_step(AdamParams(weight_decay=0.1, grad_clip_norm=1.0), learning_rate=3e-5).result()
        sent = fake.requests[-2].body["adam_params"]
        assert sent["learning_rate"] == 3e-5 and sent["weight_decay"] == 0.1 and sent["grad_clip_norm"] == 1.0
        assert out.step == 1 and out.learning_rate == 3e-5

    def test_repr_and_info(self, project: endor.Project) -> None:
        run = project.runs.create("jevk5-4b", name="r", tags=["a"], user_metadata={"k": 1})
        assert run.id in repr(run) and "jevk5-4b" in repr(run)
        info = run.info()
        assert info.name == "r" and info.tags == ["a"] and info.user_metadata == {"k": 1} and info.lora.rank == 16


class TestDashboard:
    def test_log_and_metrics(self, project: endor.Project) -> None:
        run = project.runs.create("jevk5-4b")
        run.forward_backward(datums()).result()
        run.optim_step().result()
        run.log({"heldout/accuracy": 0.9})  # defaults to the handle's last known step
        run.log({"heldout/accuracy": 0.95, "other": 1.0}, step=1)
        points = run.metrics()
        assert {p.key for p in points} == {"train/loss", "heldout/accuracy", "other"}
        assert [p.step for p in run.metrics(keys=["heldout/accuracy"])] == [0, 1]
        assert [p.value for p in run.metrics(keys=["heldout/accuracy"], since_step=1)] == [0.95]
        auto = [p for p in points if p.key == "train/loss"]
        assert auto and auto[0].source == "auto"

    def test_log_eval(self, project: endor.Project) -> None:
        run = project.runs.create("jevk5-4b")
        ev = run.log_eval("jevk5-4b", {"accuracy": 0.5}, step=3, name="heldout")
        assert ev.source == "client" and ev.training_run_id == run.id and ev.results
        assert ev.results["accuracy"] == 0.5 and ev.results["step"] == 3
        assert [e.id for e in project.evaluations(run_id=run.id)] == [ev.id]
