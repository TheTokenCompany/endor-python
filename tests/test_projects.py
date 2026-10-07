from __future__ import annotations

import logging

import pytest

import endor
from endor import (
    ConflictError,
    EndorClient,
    LimitReachedError,
    NoBaseModelError,
    NotFoundError,
    UnprocessableEntityError,
)

from .conftest import ANGER, DEPT, requests_to, rows, unique, wait_closed
from .fake_api import FakeEndor


class TestProjects:
    def test_create_get_list_delete(self, client: EndorClient) -> None:
        name = unique("proj")
        p = client.projects.create(name, description="tickets")
        assert p.name == name and p.info_.description == "tickets" and repr(p) == f"Project({name!r})"
        assert client.projects.get(name).info_.n_runs == 0
        assert name in [x.name for x in client.projects.list()]
        with pytest.raises(ConflictError) as e:
            client.projects.create(name)
        assert e.value.code == "conflict"
        p.delete()
        with pytest.raises(NotFoundError):
            client.projects.get(name)

    def test_get_or_create(self, client: EndorClient, fake: FakeEndor) -> None:
        name = unique("goc")
        a = client.projects.get_or_create(name, base_model="decider-2b")
        b = client.projects.get_or_create(name, base_model="jev-9b")  # an existing project is left as it is
        assert a.name == b.name and len(fake.projects) == 1 and b.info_.base_model == "decider-2b"

    def test_create_with_base_model(self, client: EndorClient, fake: FakeEndor) -> None:
        name = unique("base")
        p = client.projects.create(name, base_model="decider-2b")
        assert fake.requests[-1].body == {"name": name, "description": None, "base_model": "decider-2b"}
        info = p.info_
        assert info.base_model == "decider-2b" and info.live_model == f"{name}/base"
        assert info.auto_promote and info.base_keep_warm  # a new project keeps its base warm
        assert info.n_models == 1  # the base model counts
        with pytest.raises(UnprocessableEntityError) as e:
            client.projects.create(unique("base"), base_model="gpt-9")
        assert e.value.code == "unknown_model"

    def test_without_base_model_the_first_run_sets_it(self, project: endor.Project) -> None:
        assert project.info_.base_model is None and project.info_.live_model is None
        assert project.models.list() == []
        with project.runs.create("jev-9b") as run:
            assert run.info_.source == "sdk"
        info = project.info()
        assert info.base_model == "jev-9b" and info.live_model == f"{project.name}/base"

    def test_update_base_model_auto_promote_and_base_keep_warm(self, client: EndorClient, fake: FakeEndor) -> None:
        p = client.projects.create(unique("upd"))
        with pytest.raises(NoBaseModelError) as e:
            p.update(base_keep_warm=True)
        assert e.value.code == "no_base_model" and e.value.status == 409
        info = p.update(base_model="jev-9b", auto_promote=False)
        assert fake.requests[-1].body == {"base_model": "jev-9b", "auto_promote": False}
        assert info.base_model == "jev-9b" and not info.auto_promote and info.live_model == f"{p.name}/base"
        info = p.update(base_keep_warm=False)
        assert fake.requests[-1].body == {"base_keep_warm": False} and not info.base_keep_warm
        assert p.update(base_keep_warm=True).base_keep_warm
        assert p.update(auto_promote=True).auto_promote
        with pytest.raises(ValueError, match="base_keep_warm"):
            p.update()

    def test_project_limit(self, client: EndorClient, fake: FakeEndor) -> None:
        for _ in range(7):
            client.projects.create(unique("lim"))
        with pytest.raises(LimitReachedError) as e:
            client.projects.create(unique("lim"))
        assert isinstance(e.value, ConflictError) and e.value.code == "limit_reached"

    def test_bad_name(self, client: EndorClient) -> None:
        with pytest.raises(UnprocessableEntityError) as e:
            client.projects.create("Bad Name")
        assert e.value.param == "name"

    def test_delete_with_active_run_conflicts(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            with pytest.raises(ConflictError) as e:
                project.delete()
            assert e.value.code == "run_active"
        wait_closed(run)
        project.delete()

    def test_continuous_learning_settings(self, client: EndorClient) -> None:
        name = unique("cl")
        try:
            p = client.projects.create(name, continuous_learning={"enabled": True, "base_model": "decider-2b"})
        except UnprocessableEntityError as e:
            pytest.xfail(f"waiting on the API's continuous-learning settings: {e}")
        if p.info_.continuous_learning is None:
            pytest.xfail("waiting on the API's continuous-learning settings (not in the project response)")
        cl = p.info_.continuous_learning
        assert cl.enabled and cl.base_model == "decider-2b"
        info = p.update(continuous_learning=endor.ContinuousLearning(enabled=True, base_model="jev-9b"))
        assert info.continuous_learning and info.continuous_learning.base_model == "jev-9b"
        assert info.continuous_learning.model is None  # a new base starts from scratch
        info = p.update(description="tickets", continuous_learning={"enabled": False})
        assert info.description == "tickets" and info.continuous_learning
        assert not info.continuous_learning.enabled and info.continuous_learning.base_model == "jev-9b"
        with pytest.raises(ValueError):
            p.update()

    def test_continuous_learning_wire_form(self, client: EndorClient, fake: FakeEndor) -> None:
        p = client.projects.create(unique("cl"))
        assert fake.requests[-1].body == {"name": p.name, "description": None}  # not sent unless given
        p.update(continuous_learning=endor.ContinuousLearning(base_model="decider-2b"))
        assert fake.requests[-1].method == "PATCH" and fake.requests[-1].path == f"/v1/projects/{p.name}"
        assert fake.requests[-1].body == {"continuous_learning": {"enabled": False, "base_model": "decider-2b"}}

    def test_info_refreshes_counts(self, project: endor.Project) -> None:
        project.datasets.upload("d", rows(1))
        assert project.info_.n_datasets == 0 and project.info().n_datasets == 1


class TestDatasets:
    def test_upload_list_get_rows_delete(self, project: endor.Project) -> None:
        d = project.datasets.upload("train", rows(12))
        assert d.n_rows == 12 and d.n_labelled_questions == 24 and d.question_types == {"choice": 12, "noul": 12}
        assert [x.name for x in project.datasets.list()] == ["train"]
        assert project.datasets.get("train").n_rows == 12
        got = list(project.datasets.rows("train", page=5))
        assert len(got) == 12 and got[0].id == "r0" and got[0].labels == {"dept": "tech", "urgent": False}
        assert isinstance(got[0], endor.DecisionRow)
        project.datasets.delete("train")
        with pytest.raises(NotFoundError):
            project.datasets.get("train")

    def test_rows_paging_requests(self, project: endor.Project, fake: FakeEndor) -> None:
        project.datasets.upload("d", rows(7))
        list(project.datasets.rows("d", page=3))
        pages = [r.params for r in fake.requests if r.path.endswith("/rows")]
        assert pages == [{"limit": "3", "offset": "0"}, {"limit": "3", "offset": "3"}, {"limit": "3", "offset": "6"}]

    def test_accepts_other_row_shapes(self, project: endor.Project, fake: FakeEndor) -> None:
        mixed = [
            {"state": "policy", "question": {"type": "noul"}, "expected": "yes"},
            {"state": "x", "question": {"type": "choice", "criteria": {"a": None, "b": None}}, "target": [0.3, 0.7]},
            endor.DecisionRow(state="y", questions={"s": ANGER}, labels={"s": 2}),
        ]
        d = project.datasets.upload("mixed", mixed)
        assert d.n_labelled_questions == 3
        sent = fake.requests[-1].body["rows"]
        assert sent[0]["labels"] == {"decision": True} and sent[1]["labels"] == {"decision": [0.3, 0.7]}

    def test_local_limits(self, project: endor.Project) -> None:
        with pytest.raises(ValueError, match="no rows"):
            project.datasets.upload("empty", [])
        with pytest.raises(ValueError, match="50000"):
            project.datasets.upload("huge", ({"state": "x", "questions": {"d": DEPT}} for _ in range(50_001)))

    def test_bad_row_names_its_index(self, project: endor.Project) -> None:
        bad = rows(3) + [{"state": "x", "questions": {"s": ANGER}, "labels": {"s": 9}}]
        with pytest.raises(UnprocessableEntityError) as e:
            project.datasets.upload("bad", bad)
        assert e.value.code == "invalid_row" and "rows[3]" in str(e.value)

    def test_duplicate_name(self, project: endor.Project) -> None:
        project.datasets.upload("d", rows(1))
        with pytest.raises(ConflictError):
            project.datasets.upload("d", rows(1))


class TestRunsResource:
    def test_create_sends_capture(self, project: endor.Project, fake: FakeEndor) -> None:
        run = project.runs.create(
            "jev-9b", rank=8, alpha=16.0, seed=3, train_mlp=False, name="r", tags=["t"], config={"lr": 1e-4}
        )
        body = fake.requests[-1].body
        assert body["lora"] == {
            "rank": 8,
            "alpha": 16.0,
            "seed": 3,
            "train_attn": True,
            "train_mlp": False,
            "train_readout": False,
        }
        assert (
            body["config"]["lr"] == 1e-4
            and body["config"]["base_model"] == "jev-9b"
            and body["config"]["lora"]["rank"] == 8
        )
        assert "code_hash" in body
        assert run.info_.lora.rank == 8 and run.info_.status == "ready"
        run.close()

    def test_from_model_sends_only_explicit_lora_settings(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b", rank=8) as run:
            run.save_checkpoint("v1").result()
        with project.runs.create(from_model="v1") as resumed:
            assert requests_to(fake, "/runs", "POST")[-1].body["lora"] == {}
            assert resumed.info_.lora.rank == 8  # inherited
        with project.runs.create(from_model="v1", rank=8, seed=7):
            assert requests_to(fake, "/runs", "POST")[-1].body["lora"] == {"rank": 8, "seed": 7}
        with pytest.raises(UnprocessableEntityError):
            project.runs.create(from_model="v1", rank=4)
        with project.runs.create("jev-9b"):
            lora = requests_to(fake, "/runs", "POST")[-1].body["lora"]
        assert lora == {
            "rank": 16,
            "alpha": 32.0,
            "seed": None,
            "train_attn": True,
            "train_mlp": True,
            "train_readout": False,
        }

    def test_capture_off(self, fake: FakeEndor) -> None:
        import httpx

        c = EndorClient(
            api_key="edk_test",
            base_url="https://endor.test",
            transport=httpx.MockTransport(fake.handler),
            capture=False,
        )
        p = c.projects.create("nocap")
        with p.runs.create("jev-9b", config={"lr": 1}):
            body = fake.requests[-1].body
        assert body["config"] == {"lr": 1} and "code_hash" not in body

    def test_requires_a_model(self, project: endor.Project) -> None:
        with pytest.raises(ValueError):
            project.runs.create()

    def test_unknown_base(self, project: endor.Project) -> None:
        with pytest.raises(UnprocessableEntityError) as e:
            project.runs.create("gpt-9")
        assert e.value.code == "unknown_model"

    def test_resume_from_model(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.forward_backward(endor.data.rows_to_datums(rows(2))).result()
            run.optim_step().result()
            run.save_checkpoint("v1", include_optimizer=True).result()
            run.save_checkpoint("v2").result()
        with project.runs.create(from_model="v1", include_optimizer=True) as resumed:
            assert resumed.info_.parent_model == f"{project.name}/v1" and resumed.info_.base_model == "jev-9b"
        with project.runs.create(from_model=f"{project.name}/v2") as full:
            assert full.info_.parent_model == f"{project.name}/v2"
        assert project.models.get("v1").parent_model is None
        with pytest.raises(ConflictError) as e:
            project.runs.create(from_model="v2", include_optimizer=True)
        assert e.value.code == "invalid_state"
        with pytest.raises(UnprocessableEntityError) as e2:
            project.runs.create(from_model="missing")
        assert e2.value.code == "unknown_model"

    def test_list_and_get(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b", tags=["x"]) as a, project.runs.create("jev-9b"):
            assert [r.id for r in project.runs.list(tag="x")] == [a.id]
            assert len(project.runs.list()) == 2 and len(project.runs.list(limit=1)) == 1
            assert project.runs.get(a.id).info_.tags == ["x"]

    def test_interrupted_provisioning_closes_the_run(
        self, project: endor.Project, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake.provisioning_polls = 10**6

        def interrupted(run: endor.Run, log_every: float = 30.0) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(endor.projects, "_wait_ready", interrupted)
        with pytest.raises(KeyboardInterrupt):
            project.runs.create("jev-9b")
        (run,) = fake.runs.values()
        assert run["status"] == "closed"

    def test_open_run_limit_is_not_retried(self, project: endor.Project, fake: FakeEndor) -> None:
        runs = [project.runs.create("jev-9b") for _ in range(4)]
        try:
            with pytest.raises(LimitReachedError) as e:
                project.runs.create("jev-9b")
            assert isinstance(e.value, ConflictError) and e.value.status == 409 and e.value.code == "limit_reached"
            assert all(r.id in str(e.value) for r in runs)  # the message lists the open runs
            assert len(requests_to(fake, "/runs", "POST")) == 5  # no retry
        finally:
            for r in runs:
                r.close()

    def test_wait_for_provisioning_logs_progress(
        self, project: endor.Project, fake: FakeEndor, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake.provisioning_polls = 2
        with caplog.at_level(logging.INFO, logger="endor"):
            run = project.runs.create("pplx-decider-v1.1-27b")
        messages = [r.getMessage() for r in caplog.records]
        assert any("provisioning a trainer for pplx-decider-v1.1-27b" in m for m in messages)
        assert any("ready after" in m for m in messages)
        assert run.ready.done() and run.info_.status == "ready"
        assert len([r for r in fake.requests if r.path == f"/v1/futures/{run.info_.ready_future_id}"]) == 3
        run.close()

    def test_no_wait(self, project: endor.Project, fake: FakeEndor) -> None:
        fake.provisioning_polls = 1
        with project.runs.create("jev-9b", wait=False) as run:
            assert not run.ready.done() and run.info_.status == "provisioning"
            run.forward(endor.data.rows_to_datums(rows(1))).result()  # ops queue behind provisioning
            run.ready.result()


class TestModels:
    def test_list_get_ttl_delete(self, project: endor.Project) -> None:
        with project.runs.create("jev-9b") as run:
            run.forward_backward(endor.data.rows_to_datums(rows(1))).result()
            run.optim_step().result()
            mid = run.save_checkpoint("v1").result()
        with project.runs.create("jev-9b") as other:
            other.save_checkpoint("v2").result()
        listed = project.models.list()
        assert [m.name for m in listed][0] == "base" and {m.name for m in listed} == {"base", "v1", "v2"}
        assert [m.name for m in project.models.list(run_id=run.id)] == ["v1"]
        m = project.models.get(mid)
        assert m.id == mid and m.step == 1 and m.training_run_id == run.id and not m.has_optimizer
        assert m.source == "sdk" and not m.live
        assert project.models.get("v1").id == mid
        assert project.models.set_ttl("v1", 3600).expires_at is not None
        assert project.models.set_ttl(mid, None).expires_at is None
        assert project.models.set_keep_warm("v1").keep_warm and project.models.get("v1").expires_at is None
        assert not project.models.set_keep_warm(mid, False).keep_warm
        project.models.delete("v1")
        with pytest.raises(NotFoundError):
            project.models.get("v1")
        assert [m.name for m in project.models.list()] == ["base", "v2"]

    def test_base_model_and_live(self, project: endor.Project, fake: FakeEndor) -> None:
        with project.runs.create("jev-9b") as run:
            run.save_checkpoint("v1").result()
        base = project.models.get("base")
        assert base.id == f"{project.name}/base" and base.source == "base" and base.live
        assert base.base_model == "jev-9b" and base.training_run_id is None
        assert project.models.get(f"{project.name}/base").id == base.id
        fake.promote(project.name, "v1")
        live = {m.name: m.live for m in project.models.list()}
        assert live == {"base": False, "v1": True}
        assert project.info().live_model == f"{project.name}/v1" and not project.info_.auto_promote
        with pytest.raises(UnprocessableEntityError):
            project.models.delete("base")

    def test_set_live(self, project: endor.Project, client: endor.EndorClient) -> None:
        with project.runs.create("jev-9b") as run:
            run.save_checkpoint("v1").result()
        info = project.set_live("v1")
        assert info.live_model == f"{project.name}/v1" and not info.auto_promote and project.info_ is info
        assert client.system_one("x", {"u": {"type": "noul"}}, model=project.name).model == f"{project.name}/v1"
        assert project.set_live(f"{project.name}/base").live_model == f"{project.name}/base"
        with pytest.raises(NotFoundError):
            project.set_live("v9")
        with pytest.raises(NoBaseModelError):
            client.projects.create(project.name + "-empty").set_live("base")

    def test_keep_warm_limit(self, project: endor.Project) -> None:
        with project.runs.create("gliner2.5-decide") as run:  # sets the base, kept warm: one of the 3 slots
            for i in range(3):
                run.save_checkpoint(f"w{i}").result()
        assert project.models.get("base").keep_warm
        for i in range(2):
            assert project.models.set_keep_warm(f"w{i}").keep_warm
        with pytest.raises(LimitReachedError) as e:
            project.models.set_keep_warm("w2")
        assert isinstance(e.value, ConflictError) and e.value.code == "limit_reached" and e.value.param == "keep_warm"
        assert not project.models.set_keep_warm("base", False).keep_warm  # the base's keep warm frees a slot
        assert not project.info().base_keep_warm
        assert project.models.set_keep_warm("w2").keep_warm
        with pytest.raises(LimitReachedError):
            project.models.set_keep_warm("base")
        for i in range(3):
            project.models.set_keep_warm(f"w{i}", False)


class TestEvaluations:
    def test_evaluate_and_list(self, project: endor.Project) -> None:
        project.datasets.upload("heldout", rows(6))
        with project.runs.create("jev-9b") as run:
            model = run.save_checkpoint("v1").result()
        ev = project.evaluate(model, "heldout", run_id=run.id).result()
        assert (
            ev.status == "completed"
            and ev.source == "server"
            and ev.dataset == "heldout"
            and ev.training_run_id == run.id
        )
        assert ev.results and 0 <= ev.results["overall"]["accuracy"] <= 1
        base_ev = project.evaluate("base", "heldout").result()  # "base" resolves in the project
        assert base_ev.model == "base"
        assert project.evaluate("v1", "heldout").result().status == "completed"  # so does a bare name
        assert project.evaluate(project.name, "heldout").result().status == "completed"  # its live model
        assert [e.id for e in project.evaluations(run_id=run.id)] == [ev.id]
        assert [e.id for e in project.evaluations(model="base")] == [base_ev.id]
        assert len(project.evaluations()) == 4
        with pytest.raises(NotFoundError):
            project.evaluate(model, "nope")
        with pytest.raises(endor.ModelRequiresProjectError):
            project.evaluate("jev-9b", "heldout")
