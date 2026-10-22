from __future__ import annotations

import pytest

import endor
from endor import EndorClient
from endor.recipes import DistillConfig, SupervisedConfig, distill, supervised

from .conftest import DEPT, requests_to, rows, unique, wait_closed
from .fake_api import FakeEndor

SHORT = {"forward_backward": "fb", "optim_step": "opt", "forward": "fwd", "save_checkpoint": "save"}


def run_calls(fake: FakeEndor) -> list[str]:
    """The recipe's run calls in the order sent (the order the trainer runs them), with "wait" where it polled
    futures (consecutive polls once)."""
    out: list[str] = []
    for r in fake.requests:
        kind = r.path.rsplit("/", 1)[-1]
        if r.method == "POST" and r.path.startswith("/v1/runs/") and kind in SHORT:
            out.append(SHORT[kind])
        elif r.path.startswith("/v1/futures") and out and out[-1] != "wait":
            out.append("wait")
    return out


def test_supervised_train(client: EndorClient, fake: FakeEndor) -> None:
    cfg = SupervisedConfig(batch_size=8, eval_every=2, model_name="final", warmup_steps=2)
    r = supervised.train(cfg, rows(40), client=client)
    assert r.base_metrics is not None and r.base_metrics["n"] == 8  # 4 held-out rows, 2 questions each
    assert r.model == "final"  # a custom model, saved inactive
    assert [h["step"] for h in r.history] == [2, 4, 6, 8, 9]  # 36 rows -> 72 datums -> 9 steps of 8
    assert r.final_metrics["n"] == 8
    evals = client.evaluations.list(run_id=r.run_id)
    assert len(evals) == len(r.history) + 1  # plus the base baseline
    run = client.runs.get(r.run_id)
    assert wait_closed(run) == "closed" and run.info().step == 9
    assert run.info_.config["batch_size"] == 8 and "replay_rows" not in run.info_.config
    lrs = [r.body["adam_params"]["learning_rate"] for r in fake.requests if r.path.endswith("/optim_step")]
    assert len(lrs) == 9 and lrs[0] < lrs[1] and lrs[-1] < lrs[1]
    assert fake.models[r.model]["step"] == 9 and fake.models[r.model]["live_at"] is None
    assert fake.runs[r.run_id]["base_model"] == cfg.base_model == "mapika/decider-2b"  # the default
    assert requests_to(fake, "/v1/runs", "POST")[-1].body["model"] == "final"
    base_calls = [x.body["model"] for x in requests_to(fake, "/v1/evaluations", "POST")]
    assert base_calls == ["mapika/decider-2b"]  # the untuned base, scored server-side in one prediction
    assert not requests_to(fake, "/v1/systemone")
    assert {e.model for e in evals} >= {"mapika/decider-2b"}


def test_supervised_sends_the_next_step_before_reading_the_last(client: EndorClient, fake: FakeEndor) -> None:
    cfg = SupervisedConfig(batch_size=4, eval_every=2, eval_base=False, model_name="m")
    r = supervised.train(cfg, rows(8), eval_rows=rows(1), client=client)  # 16 datums: 4 steps
    assert [h["step"] for h in r.history] == [2, 4]
    step = ["fb", "opt"]
    # Step 2 is sent before step 1 is read. A held-out evaluation reads every step first and runs right after its
    # step's optim_step, as before.
    held_out = ["wait", "fwd", "wait"]
    assert run_calls(fake) == step + step + held_out + step + step + held_out + ["save", "wait"]
    assert cfg.steps_in_flight == 2 and fake.runs[r.run_id]["config"]["steps_in_flight"] == 2


def test_supervised_with_one_step_in_flight_waits_for_each_step(client: EndorClient, fake: FakeEndor) -> None:
    cfg = SupervisedConfig(model_name=unique("sl"), batch_size=4, eval_every=2, eval_base=False, steps_in_flight=1)
    r = supervised.train(cfg, rows(8), eval_rows=rows(1), client=client)
    assert [h["step"] for h in r.history] == [2, 4]
    step = ["fb", "opt", "wait"]
    held_out = ["fwd", "wait"]
    assert run_calls(fake) == step + step + held_out + step + step + held_out + ["save", "wait"]
    with pytest.raises(ValueError, match="in_flight"):
        supervised.train(SupervisedConfig(steps_in_flight=0), rows(4), client=client)


def test_supervised_raises_a_failed_step_at_that_step(
    client: EndorClient, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sent = FakeEndor.forward

    def third_fails(self: FakeEndor, run: dict, body: dict, kind: str) -> dict:
        ref = sent(self, run, body, kind)
        if kind == "forward_backward" and len(requests_to(self, "/forward_backward")) == 3:
            self.fail_future(ref["future_id"], "oom", "out of memory")
        return ref

    monkeypatch.setattr(FakeEndor, "forward", third_fails)
    cfg = SupervisedConfig(model_name=unique("sl"), batch_size=4, eval_base=False, epochs=2)  # 8 steps
    with pytest.raises(endor.OperationFailedError) as e:
        supervised.train(cfg, rows(8), eval_rows=rows(1), client=client)
    assert e.value.code == "oom"
    progress = capsys.readouterr().err
    assert "step 2/8" in progress and "step 3/8" not in progress  # steps 1 and 2 were read and logged
    assert len(requests_to(fake, "/forward_backward")) == 4  # step 4 was in flight
    assert [r["status"] for r in fake.runs.values()] == ["closed"]


def test_supervised_trains_on_its_base_model(client: EndorClient, fake: FakeEndor) -> None:
    cfg = SupervisedConfig(model_name="v1", base_model="autotrust/jev-9b", batch_size=8)
    r = supervised.train(cfg, rows(10), eval_rows=rows(2), client=client)
    assert r.model == "v1" and fake.runs[r.run_id]["base_model"] == "autotrust/jev-9b"
    assert {x.body["model"] for x in requests_to(fake, "/v1/evaluations", "POST")} == {"autotrust/jev-9b"}
    client.custom_models.promote(r.model)  # live: decisions call it by its name
    assert client.system_one("x", {"d": DEPT}, model=r.model).model == "v1"


def test_supervised_refuses_a_taken_name(client: EndorClient) -> None:
    taken = client.projects.create(unique("sl"), base_model="autotrust/jev-9b").name
    with pytest.raises(endor.ConflictError):
        supervised.train(SupervisedConfig(model_name=taken, eval_base=False), rows(10), client=client)


def test_supervised_with_eval_rows_and_no_base(client: EndorClient) -> None:
    cfg = SupervisedConfig(batch_size=4, eval_base=False, epochs=2, lr_schedule="constant")  # a name made up
    r = supervised.train(cfg, rows(8), eval_rows=rows(2), client=client)
    assert r.base_metrics is None and len(r.history) == 1 and r.history[0]["step"] == 8
    assert r.model.startswith("sft-")


def test_supervised_closes_the_run_on_error(
    client: EndorClient, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*a: object, **k: object) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(endor.Run, "optim_step", broken)
    cfg = SupervisedConfig(model_name=unique("sl"), batch_size=4, eval_base=False)
    with pytest.raises(RuntimeError, match="network down"):
        supervised.train(cfg, rows(8), eval_rows=rows(1), client=client)
    assert [r["status"] for r in fake.runs.values()] == ["closed"]


def test_supervised_needs_labels(client: EndorClient) -> None:
    with pytest.raises(ValueError, match="no labeled"):
        supervised.train(SupervisedConfig(), [{"state": "s", "questions": {"d": DEPT}}] * 5, client=client)


def test_supervised_warns_about_options_without_labels(client: EndorClient, capsys: pytest.CaptureFixture[str]) -> None:
    train = rows(20)
    for i, r in enumerate(train):
        r["labels"]["dept"] = "billing" if i < 3 else "tech"  # 3 billing, 17 tech
    for r in train:
        r["labels"]["urgent"] = False  # never true
    cfg = SupervisedConfig(model_name=unique("sl"), batch_size=8, eval_base=False, progress=False)
    supervised.train(cfg, train, eval_rows=rows(2), client=client)
    err = capsys.readouterr().err.splitlines()
    assert err == [
        "warning: question 'dept': under 10 training labels for billing (3)",
        "warning: question 'urgent': no training labels for true; the model can't learn an option it never sees",
    ]


def test_supervised_does_not_warn_when_every_option_has_labels(
    client: EndorClient, capsys: pytest.CaptureFixture[str]
) -> None:
    def cfg() -> SupervisedConfig:
        return SupervisedConfig(model_name=unique("sl"), batch_size=8, eval_base=False, progress=False)

    supervised.train(cfg(), rows(40), eval_rows=rows(2), client=client)
    soft = [{"state": "s", "questions": {"d": DEPT}, "labels": {"d": {"billing": 0.1, "tech": 0.9}}}] * 12
    supervised.train(cfg(), soft, eval_rows=rows(2), client=client)  # a soft label gives every option some weight
    assert capsys.readouterr().err == ""


def test_label_warnings_list_at_most_ten_options() -> None:
    many = endor.Choice(criteria={f"o{i}": None for i in range(15)})
    lines = supervised._label_warnings([endor.DecisionRow(state="s", questions={"q": many}, labels={"q": "o0"})] * 12)
    assert lines == [
        "question 'q': no training labels for o1, o2, o3, o4, o5, o6, o7, o8, o9, o10 and 4 more; "
        "the model can't learn an option it never sees"
    ]


def test_replay(client: EndorClient, fake: FakeEndor) -> None:
    replay = [{"state": {"g": i}, "questions": {"d": DEPT}} for i in range(3)]
    cfg = SupervisedConfig(
        model_name=unique("rp"), batch_size=4, eval_base=False, replay_rows=replay, replay_per_batch=2
    )
    supervised.train(cfg, rows(8), eval_rows=rows(1), client=client)
    fbs = [r.body for r in fake.requests if r.path.endswith("/forward_backward")]
    assert all(len(b["data"]) == 6 for b in fbs)  # 4 + 2 replay datums
    assert all("probs" in b["data"][-1]["target"] for b in fbs)


def test_lr_schedule() -> None:
    cfg = SupervisedConfig(learning_rate=1.0, warmup_steps=4)
    assert [supervised.lr_at(cfg, s, 8) for s in (0, 3, 4, 7)] == [0.25, 0.625, 0.5, 0.125]
    cfg.lr_schedule = "constant"
    assert supervised.lr_at(cfg, 7, 8) == 1.0


def test_evaluate_model(client: EndorClient) -> None:
    m = supervised.evaluate_model(client, "autotrust/jev-9b", rows(3))  # a base model, or a custom model live or not
    assert m["n"] == 6 and m["accuracy"] in (0.0, 0.5, 1.0) or 0 <= m["accuracy"] <= 1


def test_distill(client: EndorClient) -> None:
    class Fixed:
        calls = 0

        def label(self, rs: list[endor.DecisionRow]) -> list[dict | None]:
            self.calls += 1
            return [{"dept": {"billing": 0.6, "tech": 0.4}, "ignored": "x"} for _ in rs]

    unlabeled = [{"state": {"body": f"t{i}"}, "questions": {"dept": DEPT}} for i in range(20)]
    sl = SupervisedConfig(batch_size=8, eval_base=False, model_name="d1")
    teacher = Fixed()
    r = distill.distill(DistillConfig(sl, teacher=teacher), unlabeled, eval_rows=rows(6), client=client)
    assert r.model == "d1" and teacher.calls == 1


def test_label_with_teacher_keeps_existing_labels_and_budget() -> None:
    class T:
        def label(self, rs: list[endor.DecisionRow]) -> list[dict | None]:
            return [{"dept": "billing", "urgent": None} for _ in rs]

    mixed = [
        {"state": "a", "questions": {"dept": DEPT}},
        {"state": "b", "questions": {"dept": DEPT}, "labels": {"dept": "tech"}},
    ]
    out = distill.label_with_teacher(mixed, DistillConfig(SupervisedConfig(), teacher=T()))
    assert [r.labels for r in out] == [{"dept": "billing"}, {"dept": "tech"}]
    half = distill.label_with_teacher(mixed * 50, DistillConfig(SupervisedConfig(), teacher=T(), budget=0.5, seed=1))
    assert 20 < len(half) < 80
    with pytest.raises(ValueError):
        distill.label_with_teacher(mixed, DistillConfig(SupervisedConfig(), teacher=T(), budget=0))
