from __future__ import annotations

import pytest

import endor
from endor import EndorClient
from endor.recipes import DistillConfig, SupervisedConfig, distill, supervised

from .conftest import DEPT, rows, unique, wait_closed
from .fake_api import FakeEndor


def test_supervised_train(client: EndorClient, fake: FakeEndor) -> None:
    cfg = SupervisedConfig(project=unique("sl"), batch_size=8, eval_every=2, model_name="final", warmup_steps=2)
    r = supervised.train(cfg, rows(40), client=client)
    assert r.base_metrics is not None and r.base_metrics["n"] == 8  # 4 held-out rows, 2 questions each
    assert r.model == f"{cfg.project}/final"
    assert [h["step"] for h in r.history] == [2, 4, 6, 8, 9]  # 36 rows -> 72 datums -> 9 steps of 8
    assert r.final_metrics["n"] == 8
    project = client.projects.get(cfg.project)
    evals = project.evaluations(run_id=r.run_id)
    assert len(evals) == len(r.history) + 1  # plus the base baseline
    run = project.runs.get(r.run_id)
    assert wait_closed(run) == "closed" and run.info().step == 9
    assert run.info_.config["batch_size"] == 8 and "replay_rows" not in run.info_.config
    lrs = [r.body["adam_params"]["learning_rate"] for r in fake.requests if r.path.endswith("/optim_step")]
    assert len(lrs) == 9 and lrs[0] < lrs[1] and lrs[-1] < lrs[1]
    assert fake.models[r.model]["step"] == 9
    assert project.info_.base_model == cfg.base_model == "decider-2b"  # the default, set on the new project
    base_calls = [x.body["model"] for x in fake.requests if x.path == "/v1/systemone"]
    assert base_calls and set(base_calls) == {f"{cfg.project}/base"}  # the baseline goes through the project
    assert {e.model for e in evals} >= {f"{cfg.project}/base"}


def test_supervised_sets_the_base_of_a_project_without_one(client: EndorClient) -> None:
    name = client.projects.create(unique("sl"), base_model="jev-9b").name
    cfg = SupervisedConfig(project=name, base_model="jev-9b", batch_size=8, model_name="v1")
    supervised.train(cfg, rows(10), eval_rows=rows(2), client=client)
    assert client.projects.get(name).info_.base_model == "jev-9b"


def test_supervised_refuses_another_projects_base(client: EndorClient, fake: FakeEndor) -> None:
    name = client.projects.create(unique("sl"), base_model="decider-2b").name
    cfg = SupervisedConfig(project=name, base_model="jev-9b", batch_size=8)
    with pytest.raises(ValueError, match="decider-2b"):
        supervised.train(cfg, rows(10), eval_rows=rows(2), client=client)
    assert not fake.runs  # before any GPU is requested


def test_supervised_with_eval_rows_and_no_base(client: EndorClient) -> None:
    cfg = SupervisedConfig(project=unique("sl"), batch_size=4, eval_base=False, epochs=2, lr_schedule="constant")
    r = supervised.train(cfg, rows(8), eval_rows=rows(2), client=client)
    assert r.base_metrics is None and len(r.history) == 1 and r.history[0]["step"] == 8
    assert r.model.startswith(f"{cfg.project}/sft-")


def test_supervised_closes_the_run_on_error(
    client: EndorClient, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*a: object, **k: object) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(endor.Run, "optim_step", broken)
    cfg = SupervisedConfig(project=unique("sl"), batch_size=4, eval_base=False)
    with pytest.raises(RuntimeError, match="network down"):
        supervised.train(cfg, rows(8), eval_rows=rows(1), client=client)
    assert [r["status"] for r in fake.runs.values()] == ["closed"]


def test_supervised_needs_labels(client: EndorClient) -> None:
    with pytest.raises(ValueError, match="no labeled"):
        supervised.train(SupervisedConfig(project="x"), [{"state": "s", "questions": {"d": DEPT}}] * 5, client=client)


def test_replay(client: EndorClient, fake: FakeEndor) -> None:
    replay = [{"state": {"g": i}, "questions": {"d": DEPT}} for i in range(3)]
    cfg = SupervisedConfig(project=unique("rp"), batch_size=4, eval_base=False, replay_rows=replay, replay_per_batch=2)
    supervised.train(cfg, rows(8), eval_rows=rows(1), client=client)
    fbs = [r.body for r in fake.requests if r.path.endswith("/forward_backward")]
    assert all(len(b["data"]) == 6 for b in fbs)  # 4 + 2 replay datums
    assert all("probs" in b["data"][-1]["target"] for b in fbs)


def test_lr_schedule() -> None:
    cfg = SupervisedConfig(project="x", learning_rate=1.0, warmup_steps=4)
    assert [supervised.lr_at(cfg, s, 8) for s in (0, 3, 4, 7)] == [0.25, 0.625, 0.5, 0.125]
    cfg.lr_schedule = "constant"
    assert supervised.lr_at(cfg, 7, 8) == 1.0


def test_evaluate_model(client: EndorClient) -> None:
    name = client.projects.create(unique("ev"), base_model="jev-9b").name
    m = supervised.evaluate_model(client, f"{name}/base", rows(3))
    assert m["n"] == 6 and m["accuracy"] in (0.0, 0.5, 1.0) or 0 <= m["accuracy"] <= 1


def test_distill(client: EndorClient) -> None:
    class Fixed:
        calls = 0

        def label(self, rs: list[endor.DecisionRow]) -> list[dict | None]:
            self.calls += 1
            return [{"dept": {"billing": 0.6, "tech": 0.4}, "ignored": "x"} for _ in rs]

    unlabeled = [{"state": {"body": f"t{i}"}, "questions": {"dept": DEPT}} for i in range(20)]
    sl = SupervisedConfig(project=unique("distill"), batch_size=8, eval_base=False, model_name="d1")
    teacher = Fixed()
    r = distill.distill(DistillConfig(sl, teacher=teacher), unlabeled, eval_rows=rows(6), client=client)
    assert r.model.endswith("/d1") and teacher.calls == 1


def test_label_with_teacher_keeps_existing_labels_and_budget() -> None:
    class T:
        def label(self, rs: list[endor.DecisionRow]) -> list[dict | None]:
            return [{"dept": "billing", "urgent": None} for _ in rs]

    mixed = [
        {"state": "a", "questions": {"dept": DEPT}},
        {"state": "b", "questions": {"dept": DEPT}, "labels": {"dept": "tech"}},
    ]
    out = distill.label_with_teacher(mixed, DistillConfig(SupervisedConfig(project="x"), teacher=T()))
    assert [r.labels for r in out] == [{"dept": "billing"}, {"dept": "tech"}]
    half = distill.label_with_teacher(
        mixed * 50, DistillConfig(SupervisedConfig(project="x"), teacher=T(), budget=0.5, seed=1)
    )
    assert 20 < len(half) < 80
    with pytest.raises(ValueError):
        distill.label_with_teacher(mixed, DistillConfig(SupervisedConfig(project="x"), teacher=T(), budget=0))
