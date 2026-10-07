"""Runs: sequence numbers, chunking, lifecycle and dashboard calls."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

import endor
from endor import AdamParams, ConflictError, Datum, OperationFailedError, Target, UnprocessableEntityError

from .conftest import DEPT, rows, wait_closed
from .fake_api import FakeEndor, HTTPError


def datums(n: int = 2) -> list[Datum]:
    return endor.data.rows_to_datums(rows(n))


def seq_ids(fake: FakeEndor, op: str) -> list[int]:
    return [r.body["seq_id"] for r in fake.requests if r.method == "POST" and r.path.endswith(f"/{op}")]


class TestSequencing:
    def test_consecutive_sequence_numbers(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b") as run:
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
        with project.runs.create("jev-9b") as run:
            run.forward_backward(datums()).result()
            again = project.runs.get(run.id)
            assert again.next_seq_id == 1
            again.optim_step().result()
            assert again.next_seq_id == 2

    def test_two_handles_conflict(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as a:
            b = project.runs.get(a.id)
            a.forward(datums(1)).result()
            with pytest.raises(ConflictError) as e:
                b.forward(datums(2))
            assert e.value.code == "seq_conflict" and e.value.param == "seq_id" and "expected" in str(e.value)
            assert b.next_seq_id == 0  # a rejected call does not advance the handle

    def test_transport_retry_reuses_seq_id_and_body(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b") as run:
            fake.fail_next.append(httpx.ConnectError("reset"))
            out = run.forward_backward(datums(3)).result()
        posts = [r for r in fake.requests if r.path.endswith("/forward_backward")]
        assert len(posts) == 2 and posts[0].body == posts[1].body and posts[1].body["seq_id"] == 0
        assert out.metrics["n"] == 6 and run.next_seq_id == 1

    def test_server_side_idempotent_retry(self, project: endor.Project) -> None:
        """The first POST reached the server (accepted) but the response was lost: the retry gets the same future."""
        with project.runs.create("jev-9b") as run:
            batch = datums(1)
            fut = run.forward(batch)
            fut.result()
            assert run.info().next_seq_id == 1
            # A literal duplicate of an accepted body is answered with the existing future, not a 409.
            body = {"data": [d.to_wire() for d in batch], "loss_fn": "cross_entropy", "seq_id": 0}
            ref = run._t.request("POST", f"/v1/runs/{run.id}/forward", json=body)
            assert ref["future_id"] == fut.id

    def test_bad_datum_is_caught_before_sending(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b") as run:
            bad = Datum(state="x", question=DEPT, target=Target(label="nope"))
            with pytest.raises(ValueError, match=r"data\[2\]: target.label 'nope'"):
                run.forward_backward(datums(1) + [bad])
            assert run.next_seq_id == 0 and seq_ids(fake, "forward_backward") == []
            with pytest.raises(ValueError, match="sum to 1"):
                run.forward([Datum(state="x", question=DEPT, target=Target(probs={"billing": 0.5, "tech": 0.1}))])
            run.forward_backward(datums()).result()
            assert seq_ids(fake, "forward_backward") == [0]

    def test_rejected_call_does_not_consume_a_number(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b") as run:
            fake.blocked = True
            with pytest.raises(endor.InsufficientBalanceError) as e:
                run.forward_backward(datums())
            assert e.value.code == "insufficient_balance" and e.value.status == 402
            assert run.next_seq_id == 0 and len(seq_ids(fake, "forward_backward")) == 1  # 402 is not retried
            fake.blocked = False
            run.forward_backward(datums()).result()
            assert seq_ids(fake, "forward_backward") == [0, 0]

    def test_refused_chunk_cancels_the_accepted_ones(
        self, project: endor.Project, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(endor.runs, "MAX_DATUMS_PER_CALL", 2)
        fake.future_polls = 5  # accepted chunks stay pending, so they can be cancelled
        with project.runs.create("jev-9b") as run:
            first = run.forward_backward  # chunk 1 is accepted, chunk 2 is refused
            fake.fail_next += [None, HTTPError(402, "insufficient_balance", "used up")]
            with pytest.raises(endor.InsufficientBalanceError):
                first(datums(2))
            cancels = [r.path for r in fake.requests if r.path.endswith("/cancel")]
            assert cancels == ["/v1/futures/fut_0002/cancel"] and fake.futures["fut_0002"]["status"] == "cancelled"
            assert run.next_seq_id == 1

    def test_closed_run_rejects_ops(self, project: endor.Project) -> None:
        run = project.runs.create("jev-9b")
        run.close()
        with pytest.raises(ConflictError) as e:
            run.forward(datums(1))
        assert e.value.code == "invalid_state"

    async def test_async_calls_are_serialized(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create("jev-9b")
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
        assert (await run.close_async()).status in ("closing", "closed", "ready")
        assert wait_closed(run) == "closed"


class TestChunking:
    def test_big_batches_are_split_and_merged(
        self, project: endor.Project, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(endor.runs, "MAX_DATUMS_PER_CALL", 7)
        with project.runs.create("jev-9b") as run:
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
        with project.runs.create("jev-9b") as run, pytest.raises(ValueError, match="no data"):
            run.forward([])


class TestLifecycle:
    def test_with_block_closes(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.forward_backward(datums())
        assert wait_closed(run) == "closed"

    def test_close_wait(self, project: endor.Project) -> None:
        run = project.runs.create("jev-9b")
        run.forward(datums(1))
        assert run.close(wait=True, timeout=30).status == "closed"

    def test_with_block_closes_on_error(self, project: endor.Project) -> None:
        with pytest.raises(RuntimeError), project.runs.create("jev-9b") as run:
            raise RuntimeError("user code failed")
        assert wait_closed(run) == "closed"

    async def test_async_with_block_closes(self, project: endor.Project) -> None:
        async with project.runs.create("jev-9b") as run:
            await (await run.forward_async(datums(1))).result_async()
        assert wait_closed(run) == "closed"

    def test_close_warns_about_unsaved_steps(self, project: endor.Project, caplog: pytest.LogCaptureFixture) -> None:
        run = project.runs.create("jev-9b")
        run.forward_backward(datums()).result()
        run.optim_step().result()
        with caplog.at_level(logging.WARNING, logger="endor"):
            run.close()
        assert any("newer than the last save" in r.getMessage() for r in caplog.records)
        wait_closed(run)

    def test_close_is_quiet_after_a_save(self, project: endor.Project, caplog: pytest.LogCaptureFixture) -> None:
        run = project.runs.create("jev-9b")
        run.forward_backward(datums()).result()
        run.optim_step().result()
        run.save_checkpoint("v1").result()
        with caplog.at_level(logging.WARNING, logger="endor"):
            run.close()
        assert not caplog.records
        fresh = project.runs.create("jev-9b")
        with caplog.at_level(logging.WARNING, logger="endor"):
            fresh.close()
        assert not caplog.records

    def test_failed_op(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b") as run:
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

    def test_trainer_lost_hint_is_not_repeated(self) -> None:
        e = OperationFailedError(
            "The training GPU was lost. Start a new run from your last saved model.", "trainer_lost"
        )
        assert str(e).lower().count("new run") == 1

    def test_save_conflict_and_options(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.save_checkpoint("v1", include_optimizer=True, ttl_seconds=7200, user_metadata={"note": "x"}).result()
            m = project.models.get("v1")
            assert m.has_optimizer and m.expires_at is not None and m.user_metadata == {"note": "x"}
            with pytest.raises(ConflictError) as e:
                run.save_checkpoint("v1")
            assert e.value.code == "conflict"
            with pytest.raises(UnprocessableEntityError):
                run.save_checkpoint("v2", ttl_seconds=10)

    def test_optim_step_params(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b") as run:
            run.forward_backward(datums()).result()
            out = run.optim_step(AdamParams(weight_decay=0.1, grad_clip_norm=1.0), learning_rate=3e-5).result()
            sent = fake.requests[-2].body["adam_params"]
            assert sent["learning_rate"] == 3e-5 and sent["weight_decay"] == 0.1 and sent["grad_clip_norm"] == 1.0
            assert out.step == 1 and out.learning_rate == 3e-5

    def test_optim_step_without_gradients_fails(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            with pytest.raises(OperationFailedError) as e:
                run.optim_step().result()
            assert e.value.code == "no_gradients" and not e.value.retryable

    def test_repr_and_info(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b", name="r", tags=["a"], user_metadata={"k": 1}) as run:
            assert run.id in repr(run) and "jev-9b" in repr(run)
            info = run.info()
        assert info.name == "r" and info.tags == ["a"] and info.user_metadata == {"k": 1} and info.lora.rank == 16
        assert info.failure is None


class TestDashboard:
    def test_log_and_metrics(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.forward_backward(datums()).result()
            run.optim_step().result()
            run.log({"heldout/accuracy": 0.9})  # no step: the run's current step on the server (1)
            run.log({"heldout/accuracy": 0.95, "other": 1.0}, step=2)
            points = run.metrics()
            assert {"train/loss", "train/lr", "heldout/accuracy", "other"} <= {p.key for p in points}
            assert [p.step for p in run.metrics(keys=["heldout/accuracy"])] == [1, 2]
            assert [p.value for p in run.metrics(keys=["heldout/accuracy"], since_step=2)] == [0.95]
            auto = [p for p in points if p.key == "train/loss"]
            assert auto and auto[0].source == "auto"

    def test_log_eval(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            ev = run.log_eval("jev-9b", {"accuracy": 0.5}, step=3, name="heldout")
        assert ev.source == "client" and ev.training_run_id == run.id and ev.results
        assert ev.results["accuracy"] == 0.5 and ev.results["step"] == 3
        assert [e.id for e in project.evaluations(run_id=run.id)] == [ev.id]
