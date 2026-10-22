from __future__ import annotations

import logging
from pathlib import Path

import pytest

import endor
from endor import (
    ConflictError,
    EndorClient,
    LimitReachedError,
    ModelInProductionError,
    ModelNotInProductionError,
    NoPreviewError,
    NotFoundError,
    ProductionFullError,
    UnprocessableEntityError,
    VersionNotCallableError,
)

from .conftest import ANGER, BASE, DEPT, URGENT, requests_to, rows, unique
from .fake_api import FakeEndor

Q = {"dept": DEPT, "urgent": URGENT}


def train_one(client: EndorClient, name: str, *, include_optimizer: bool = False) -> str:
    """A custom model trained one step and saved (inactive); its name."""
    with client.custom_models.create(name, base_model=BASE) as run:
        run.forward_backward(endor.data.rows_to_datums(rows(1))).result()
        run.optim_step().result()
        return run.save_checkpoint(include_optimizer=include_optimizer).result()


class TestProjects:
    def test_create_get_list_delete(self, client: EndorClient, fake: FakeEndor) -> None:
        name = unique("proj")
        p = client.projects.create(name, base_model=BASE, description="tickets")
        assert fake.requests[-1].body == {"name": name, "description": "tickets", "base_model": BASE}
        assert "idempotency-key" in fake.requests[-1].headers
        assert p.name == name and repr(p) == f"Project({name!r})"
        info = p.info_
        assert info.base_model == BASE and info.description == "tickets" and not info.auto_promote  # off by default
        assert info.production is None and info.preview is None and info.versions == 0
        assert client.projects.get(name).info_.base_model == BASE
        assert name in [x.name for x in client.projects.list()]
        with pytest.raises(ConflictError) as e:
            client.projects.create(name, base_model=BASE)
        assert e.value.code == "conflict"
        p.delete()
        with pytest.raises(NotFoundError):
            client.projects.get(name)

    def test_get_or_create_keeps_its_one_base(self, client: EndorClient, fake: FakeEndor) -> None:
        name = unique("goc")
        a = client.projects.get_or_create(name, base_model="mapika/decider-2b")
        b = client.projects.get_or_create(name, base_model="mapika/decider-2b", description="ignored")
        assert a.name == b.name and len(fake.projects) == 1 and b.info_.description is None
        assert fake.requests[-1].method == "GET"
        n = len(fake.requests)
        with pytest.raises(ValueError, match="chosen once"):
            client.projects.get_or_create(name, base_model=BASE)
        assert [r.method for r in fake.requests[n:]] == ["GET"]

    def test_get_or_create_takes_a_bare_base_id(self, client: EndorClient, fake: FakeEndor) -> None:
        name = unique("bare")  # the API reads "decider-2b" as "mapika/decider-2b" until 2026-11-30
        a = client.projects.get_or_create(name, base_model="decider-2b")
        b = client.projects.get_or_create(name, base_model="decider-2b")
        assert a.info_.base_model == b.info_.base_model == "mapika/decider-2b" and len(fake.projects) == 1
        with pytest.raises(ValueError, match="chosen once"):
            client.projects.get_or_create(name, base_model="jev-9b")

    def test_the_release_befores_arguments_say_what_to_do(self, client: EndorClient, fake: FakeEndor) -> None:
        with pytest.raises(TypeError, match=r"one base model now: pass base_model='mapika/decider-2b'"):
            client.projects.create(unique("old"), base_models=["mapika/decider-2b"])  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="kind was removed.*custom_models.create"):
            client.projects.create(unique("old"), base_model=BASE, kind="custom")  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="kind was removed"):
            client.projects.get_or_create(unique("old"), base_model=BASE, kind="managed")  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="pass the project's base model"):
            client.projects.create(unique("old"))
        with pytest.raises(TypeError, match="unexpected keyword argument 'colour'"):
            client.projects.create(unique("old"), base_model=BASE, colour="red")  # type: ignore[call-arg]
        assert not fake.requests  # refused before sending anything

    def test_unknown_base(self, client: EndorClient) -> None:
        with pytest.raises(UnprocessableEntityError) as e:
            client.projects.create(unique("base"), base_model="gpt-9")
        assert e.value.code == "invalid_input" and e.value.param == "base_model"

    def test_update_description(self, client: EndorClient, fake: FakeEndor) -> None:
        p = client.projects.create(unique("upd"), base_model=BASE)
        info = p.update(description="tickets")
        assert fake.requests[-1].body == {"description": "tickets"} and info.description == "tickets"
        assert p.info_ is info
        with pytest.raises(ValueError, match="nothing to update"):
            p.update()

    def test_ref_answers_with_the_base_until_the_first_version(self, client: EndorClient) -> None:
        p = client.projects.create(unique("ref"), base_model=BASE)
        assert p.ref == p.name and p.preview_ref is None  # no version yet, so no preview
        assert client.system_one("x", Q, model=p.ref).model == p.name
        with pytest.raises(NoPreviewError) as e:
            client.system_one("x", Q, model=f"{p.name}:preview")
        assert isinstance(e.value, NotFoundError) and e.value.code == "no_preview"

    def test_versions_live_preview_promote_and_auto_promote(self, client: EndorClient, fake: FakeEndor) -> None:
        p = client.projects.create(unique("ver"), base_model=BASE)
        fake.add_version(p.name, accuracy=81.0, loss=0.5)  # auto-promote is off by default: v1 is the preview
        info = p.info()
        assert (info.production, info.preview, info.versions, info.auto_promote) == (None, "v1", 1, False)
        assert p.preview_ref == f"{p.name}:preview"
        assert client.system_one("x", Q, model=p.ref).model == p.name  # the base until a promote
        assert client.system_one("x", Q, model=p.preview_ref).model == f"{p.name}:v1"

        info = p.promote("v1")
        req = fake.requests[-1]
        assert (req.method, req.path) == ("POST", f"/v1/projects/{p.name}/versions/v1/promote")
        assert (info.production, info.preview, info.auto_promote) == ("v1", None, False) and p.info_ is info
        assert info.promoted_at is not None and p.preview_ref is None
        assert (p.production, p.preview, p.auto_promote) == ("v1", None, False)  # the Project's shortcuts to info_
        with pytest.raises(NoPreviewError):
            client.system_one("x", Q, model=f"{p.name}:preview")
        fake.add_version(p.name, accuracy=84.5, loss=0.4)  # v2 waits in the preview
        assert client.system_one("x", Q, model=p.ref).model == f"{p.name}:v1"
        assert client.system_one("x", Q, model=f"{p.name}:preview").model == f"{p.name}:v2"
        vs = p.versions()
        assert [(v.name, v.production, v.preview) for v in vs] == [("v2", False, True), ("v1", True, False)]
        assert vs[0].accuracy == 84.5 and vs[1].loss == 0.5 and vs[0].base_model == BASE

        info = p.set_auto_promote(True)  # the newest goes live at once: no preview any more
        assert fake.requests[-1].body == {"auto_promote": True} and p.auto_promote is True
        assert (info.production, info.preview, info.auto_promote) == ("v2", None, True)
        fake.add_version(p.name)  # with auto-promote on, v3 goes live by itself
        assert (p.info().production, p.info_.preview) == ("v3", None)
        vs = p.versions()
        assert [(v.name, v.production, v.preview) for v in vs] == [
            ("v3", True, False),
            ("v2", False, False),
            ("v1", False, False),
        ]

        info = p.promote("v1")  # a rollback: auto-promote turns off, and the newest is the preview again
        assert (info.production, info.preview, info.auto_promote) == ("v1", "v3", False)
        assert not any(v.production and v.preview for v in p.versions())
        with pytest.raises(NotFoundError):
            p.promote("v9")

    def test_older_versions_and_old_references_cant_be_called(self, client: EndorClient, fake: FakeEndor) -> None:
        p = client.projects.create(unique("old"), base_model=BASE)
        fake.add_version(p.name)
        fake.add_version(p.name)
        with pytest.raises(VersionNotCallableError) as e:
            client.system_one("x", Q, model=f"{p.name}:v1")
        assert isinstance(e.value, NotFoundError) and e.value.code == "version_not_callable"
        with pytest.raises(NotFoundError) as e2:
            client.system_one("x", Q, model=f"{p.name}/v1")  # the release before's form
        assert e2.value.code == "unknown_model"
        with pytest.raises(NotFoundError):
            client.system_one("x", Q, model=f"{p.name}:staging")
        with pytest.raises(NotFoundError) as e3:
            client.system_one("x", Q, model=BASE.partition("/")[2])  # a base model's bare id
        assert e3.value.code == "unknown_model" and BASE in e3.value.message

    def test_download_a_version(self, client: EndorClient, fake: FakeEndor, tmp_path: Path) -> None:
        p = client.projects.create(unique("dl"), base_model=BASE)
        fake.add_version(p.name)
        files = p.download("v1", tmp_path / "v1")
        assert {f.name for f in files} == {"adapter_model.safetensors", "adapter_config.json", "endor_manifest.json"}
        assert all(f.path.read_bytes() for f in files)
        assert requests_to(fake, "/versions/v1/download")[-1].params == {"include_optimizer": "false"}

    def test_learning_status(self, client: EndorClient, fake: FakeEndor) -> None:
        p = client.projects.create(unique("learn"), base_model=BASE)
        got = p.learning()
        assert got.status == "waiting" and got.serving == p.name and got.trainings == []
        fake.learning[p.name] = {
            "status": "learning",
            "trainings": [
                {
                    "at": "2026-10-20T12:00:00Z",
                    "kept": True,
                    "version": "v1",
                    "decisions": 400,
                    "accuracy": 91.5,
                    "low": 88.0,
                    "high": 94.0,
                    "loss": 0.31,
                    "against": None,
                    "against_accuracy": 84.0,
                    "against_loss": 0.52,
                    "worse": [],
                }
            ],
            "drift": [{"question": "dept", "changed": True, "answer_shift": 0.14, "confidence_change": -0.06}],
        }
        got = p.learning()
        assert got.trainings[0].decisions == 400 and got.trainings[0].version == "v1"
        assert got.drift[0].question == "dept" and got.drift[0].changed and got.drift[0].answer_shift == 0.14

    def test_project_limit(self, client: EndorClient) -> None:
        for _ in range(5):
            client.projects.create(unique("lim"), base_model=BASE)
        with pytest.raises(LimitReachedError) as e:
            client.projects.create(unique("lim"), base_model=BASE)
        assert isinstance(e.value, ConflictError) and e.value.code == "limit_reached"

    def test_bad_name(self, client: EndorClient) -> None:
        with pytest.raises(UnprocessableEntityError) as e:
            client.projects.create("Bad Name", base_model=BASE)
        assert e.value.param == "name"

    def test_projects_and_custom_models_share_names(self, client: EndorClient) -> None:
        p = client.projects.create(unique("shared"), base_model=BASE)
        with pytest.raises(ConflictError):
            client.custom_models.create(p.name, base_model=BASE)
        name = train_one(client, unique("cm"))
        with pytest.raises(ConflictError):
            client.projects.create(name, base_model=BASE)


class TestCustomModels:
    def test_create_train_save_promote_call(self, client: EndorClient, fake: FakeEndor) -> None:
        name = unique("router")
        with client.custom_models.create(name, base_model=BASE, rank=8) as run:
            body = requests_to(fake, "/v1/runs", "POST")[-1].body
            assert body["model"] == name and body["base_model"] == BASE and body["lora"]["rank"] == 8
            assert run.model == name and run.info_.model == name
            run.forward_backward(endor.data.rows_to_datums(rows(2))).result()
            run.optim_step().result()
            saved = run.save_checkpoint().result()  # no name: the run's model
            assert requests_to(fake, "/save_checkpoint")[-1].body["name"] is None
        assert saved == name
        m = client.custom_models.get(name)
        assert (m.name, m.base_model, m.production, m.state, m.promoted_at, m.step) == (
            name,
            BASE,
            False,
            "archived",
            None,
            1,
        )  # every save starts in the archive
        assert m.training_run_id == run.id and m.loss is not None
        with pytest.raises(ModelNotInProductionError) as e:
            client.system_one("x", Q, model=name)
        assert isinstance(e.value, ConflictError) and e.value.code == "model_not_in_production"
        prod = client.custom_models.promote(name)
        assert prod.production and prod.state == "production" and prod.promoted_at is not None
        assert requests_to(fake, f"/custom_models/{name}/promote")[-1].body == {}
        assert client.system_one("x", Q, model=name).model == name
        assert client.custom_models.promote(name).production  # promoting it again changes nothing
        archived = client.custom_models.archive(name)
        assert (archived.production, archived.state) == (False, "archived")
        assert requests_to(fake, f"/custom_models/{name}/demote", "POST")  # the archive route
        with pytest.raises(ModelNotInProductionError):
            client.system_one("x", Q, model=name)
        assert not client.custom_models.archive(name).production  # nor archiving it again

    def test_serving_state(self, client: EndorClient, fake: FakeEndor) -> None:
        """In production, ``serving`` says whether a decision server has the model loaded; None while archived."""
        name = train_one(client, unique("srv"))
        archived = client.custom_models.get(name)
        assert (archived.serving, archived.serving_error) == (None, None)
        assert client.custom_models.promote(name).serving == "live"
        fake.models[name]["_serving"], fake.models[name]["_serving_error"] = "failed", "the adapter didn't load"
        failed = client.custom_models.get(name)
        assert (failed.serving, failed.serving_error) == ("failed", "the adapter didn't load")
        assert [(m.name, m.serving) for m in client.custom_models.list()] == [(name, "failed")]
        again = client.custom_models.promote(name)  # promoting a failed one deploys it again
        assert (again.serving, again.serving_error) == ("live", None)
        fake.models[name]["_serving"] = "deploying"
        assert client.custom_models.get(name).serving == "deploying"
        assert client.custom_models.archive(name).serving is None
        unknown = endor.CustomModelInfo.model_validate(
            {**fake.model_view(fake.models[name]), "serving": "warming", "serving_error": None}
        )
        assert unknown.serving == "warming"  # a value a newer server sends doesn't break the SDK

    def test_each_save_is_a_new_model(self, client: EndorClient) -> None:
        name = unique("cm")
        with client.custom_models.create(name, base_model=BASE) as run:
            run.save_checkpoint().result()
            with pytest.raises(ConflictError):
                run.save_checkpoint().result()  # the name is taken now
            other = run.save_checkpoint(f"{name}-b").result()
        assert other == f"{name}-b" and len(client.custom_models.list()) == 2
        with client.runs.create(BASE) as unnamed, pytest.raises(UnprocessableEntityError):
            unnamed.save_checkpoint().result()  # no name and no model on the run
        with pytest.raises(ValueError, match="bad model name"):
            client.custom_models.create("Bad Name", base_model=BASE)
        with pytest.raises(TypeError, match="first argument"):
            client.custom_models.create(name, base_model=BASE, model="x")

    def test_list_puts_production_ones_first(self, client: EndorClient, fake: FakeEndor) -> None:
        a, b, c = (train_one(client, unique(x)) for x in ("a", "b", "c"))
        client.custom_models.promote(a)
        names = [m.name for m in client.custom_models.list()]
        assert names[0] == a and set(names[1:]) == {b, c}
        assert [m.name for m in client.custom_models.list(production=True)] == [a]
        assert {m.name for m in client.custom_models.list(production=False)} == {b, c}
        assert requests_to(fake, "/custom_models", "GET")[-1].params["production"] == "false"
        assert len(client.custom_models.list(limit=2)) == 2

    def test_production_slots_and_replace(self, client: EndorClient, fake: FakeEndor) -> None:
        names = [train_one(client, unique(f"m{i}")) for i in range(4)]
        for n in names[:3]:
            client.custom_models.promote(n)
        with pytest.raises(ProductionFullError) as e:
            client.custom_models.promote(names[3])
        assert isinstance(e.value, LimitReachedError) and e.value.code == "production_full"
        assert all(n in str(e.value) for n in names[:3])
        assert len(requests_to(fake, f"/custom_models/{names[3]}/promote")) == 1  # never retried
        got = client.custom_models.promote(names[3], replace=names[0])
        assert got.production and requests_to(fake, f"/{names[3]}/promote")[-1].body == {"replace": names[0]}
        assert client.custom_models.get(names[0]).state == "archived"
        with pytest.raises(NotFoundError):
            client.custom_models.promote(names[0], replace=names[0])  # not in production

    def test_ttl_and_delete_refuse_a_production_model(self, client: EndorClient) -> None:
        name = train_one(client, unique("ttl"))
        assert client.custom_models.set_ttl(name, 3600).expires_at is not None
        assert client.custom_models.set_ttl(name, None).expires_at is None
        client.custom_models.promote(name)
        with pytest.raises(ModelInProductionError):
            client.custom_models.set_ttl(name, 3600)
        with pytest.raises(ModelInProductionError) as e:
            client.custom_models.delete(name)
        assert e.value.code == "model_in_production"
        client.custom_models.archive(name)
        client.custom_models.delete(name)
        with pytest.raises(NotFoundError):
            client.custom_models.get(name)

    def test_models_list_names_what_decisions_can_call(self, client: EndorClient, fake: FakeEndor) -> None:
        p = client.projects.create(unique("cat"), base_model=BASE)
        live, inactive = train_one(client, unique("prod")), train_one(client, unique("idle"))
        client.custom_models.promote(live)
        got = {m.name: m.kind for m in client.models.list().models}
        assert got == {p.name: "project", live: "custom"}  # no preview: the project has no version that isn't live
        assert inactive not in got
        fake.add_version(p.name)
        p.set_auto_promote(False)
        fake.add_version(p.name)  # v2 waits as the preview
        assert client.models.list().models[1].name == f"{p.name}:preview"


class TestDatasets:
    def test_upload_list_get_rows_delete(self, client: EndorClient, fake: FakeEndor) -> None:
        d = client.datasets.upload("train", rows(12))
        assert requests_to(fake, "/v1/datasets", "POST")[-1].path == "/v1/datasets"
        assert d.n_rows == 12 and d.n_labelled_questions == 24 and d.question_types == {"choice": 12, "noul": 12}
        assert [x.name for x in client.datasets.list()] == ["train"]
        assert client.datasets.get("train").n_rows == 12
        got = list(client.datasets.rows("train", page=5))
        assert len(got) == 12 and got[0].id == "r0" and got[0].labels == {"dept": "tech", "urgent": False}
        assert isinstance(got[0], endor.DecisionRow)
        client.datasets.delete("train")
        with pytest.raises(NotFoundError):
            client.datasets.get("train")

    def test_rows_paging_requests(self, client: EndorClient, fake: FakeEndor) -> None:
        client.datasets.upload("d", rows(7))
        list(client.datasets.rows("d", page=3))
        pages = [r.params for r in fake.requests if r.path.endswith("/rows")]
        assert pages == [{"limit": "3", "offset": "0"}, {"limit": "3", "offset": "3"}, {"limit": "3", "offset": "6"}]

    def test_accepts_other_row_shapes(self, client: EndorClient, fake: FakeEndor) -> None:
        mixed = [
            {"state": "policy", "question": {"type": "noul"}, "expected": "yes"},
            {"state": "x", "question": {"type": "choice", "criteria": {"a": None, "b": None}}, "target": [0.3, 0.7]},
            endor.DecisionRow(state="y", questions={"s": ANGER}, labels={"s": 2}),
        ]
        d = client.datasets.upload("mixed", mixed)
        assert d.n_labelled_questions == 3
        sent = fake.requests[-1].body["rows"]
        assert sent[0]["labels"] == {"decision": True} and sent[1]["labels"] == {"decision": [0.3, 0.7]}

    def test_local_limits(self, client: EndorClient) -> None:
        with pytest.raises(ValueError, match="no rows"):
            client.datasets.upload("empty", [])
        with pytest.raises(ValueError, match="50000"):
            client.datasets.upload("huge", ({"state": "x", "questions": {"d": DEPT}} for _ in range(50_001)))

    def test_bad_row_names_its_index(self, client: EndorClient) -> None:
        bad = rows(3) + [{"state": "x", "questions": {"s": ANGER}, "labels": {"s": 9}}]
        with pytest.raises(UnprocessableEntityError) as e:
            client.datasets.upload("bad", bad)
        assert e.value.code == "invalid_row" and "rows[3]" in str(e.value)

    def test_duplicate_name(self, client: EndorClient) -> None:
        client.datasets.upload("d", rows(1))
        with pytest.raises(ConflictError):
            client.datasets.upload("d", rows(1))


class TestRunsResource:
    def test_create_sends_capture(self, client: EndorClient, fake: FakeEndor) -> None:
        run = client.runs.create(
            BASE, rank=8, alpha=16.0, seed=3, train_mlp=False, name="r", tags=["t"], config={"lr": 1e-4}
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
        assert body["config"]["lr"] == 1e-4 and body["config"]["base_model"] == BASE
        assert body["config"]["lora"]["rank"] == 8 and body["model"] is None
        assert "code_hash" in body
        assert run.info_.lora.rank == 8 and run.info_.status == "ready" and run.model is None
        run.close()

    def test_needs_a_base_model(self, client: EndorClient, fake: FakeEndor) -> None:
        with pytest.raises(ValueError, match="base model"):
            client.runs.create()
        assert not fake.requests

    def test_from_model_sends_only_explicit_lora_settings(self, client: EndorClient, fake: FakeEndor) -> None:
        name = unique("v")
        with client.runs.create(BASE, rank=8) as run:
            run.save_checkpoint(name).result()
        with client.runs.create(from_model=name) as resumed:
            assert requests_to(fake, "/runs", "POST")[-1].body["lora"] == {}
            assert resumed.info_.lora.rank == 8  # inherited
        with client.runs.create(from_model=name, rank=8, seed=7):
            assert requests_to(fake, "/runs", "POST")[-1].body["lora"] == {"rank": 8, "seed": 7}
        with pytest.raises(UnprocessableEntityError):
            client.runs.create(from_model=name, rank=4)
        with client.runs.create(BASE):
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
        with c.runs.create(BASE, config={"lr": 1}):
            body = fake.requests[-1].body
        assert body["config"] == {"lr": 1} and "code_hash" not in body

    def test_resume_from_model(self, client: EndorClient) -> None:
        v1, v2 = unique("v1"), unique("v2")
        with client.runs.create(BASE) as run:
            run.forward_backward(endor.data.rows_to_datums(rows(2))).result()
            run.optim_step().result()
            run.save_checkpoint(v1, include_optimizer=True).result()
            run.save_checkpoint(v2).result()
        with client.runs.create(from_model=v1, include_optimizer=True) as resumed:
            assert resumed.info_.parent_model == v1 and resumed.info_.base_model == BASE
        assert client.custom_models.get(v1).parent_model is None
        with pytest.raises(ConflictError) as e:
            client.runs.create(from_model=v2, include_optimizer=True)
        assert e.value.code == "invalid_state"
        with pytest.raises(NotFoundError) as e2:  # 404 as for a decision's unknown model (it was 422)
            client.runs.create(from_model="missing")
        assert e2.value.code == "unknown_model"

    def test_list_and_get(self, client: EndorClient) -> None:
        with client.runs.create(BASE, tags=["x"]) as a, client.runs.create(BASE):
            assert [r.id for r in client.runs.list(tag="x")] == [a.id]
            assert len(client.runs.list()) == 2 and len(client.runs.list(limit=1)) == 1
            assert client.runs.get(a.id).info_.tags == ["x"]

    def test_interrupted_provisioning_closes_the_run(
        self, client: EndorClient, fake: FakeEndor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake.provisioning_polls = 10**6

        def interrupted(run: endor.Run, log_every: float = 30.0, progress: bool = True) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(endor.training, "_wait_ready", interrupted)
        with pytest.raises(KeyboardInterrupt):
            client.runs.create(BASE)
        (run,) = fake.runs.values()
        assert run["status"] == "closed"

    def test_open_run_limit_is_not_retried(self, client: EndorClient, fake: FakeEndor) -> None:
        runs = [client.runs.create(BASE) for _ in range(4)]
        try:
            with pytest.raises(LimitReachedError) as e:
                client.runs.create(BASE)
            assert isinstance(e.value, ConflictError) and e.value.status == 409 and e.value.code == "limit_reached"
            assert all(r.id in str(e.value) for r in runs)  # the message lists the open runs
            assert len(requests_to(fake, "/runs", "POST")) == 5  # no retry
        finally:
            for r in runs:
                r.close()

    def test_wait_for_provisioning_logs_progress(
        self, client: EndorClient, fake: FakeEndor, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake.provisioning_polls = 2
        with caplog.at_level(logging.INFO, logger="endor"):
            run = client.runs.create(BASE)
        messages = [r.getMessage() for r in caplog.records]
        assert any(f"provisioning a trainer for {BASE}" in m for m in messages)
        assert any("ready after" in m for m in messages)
        assert run.ready.done() and run.info_.status == "ready"
        assert len([r for r in fake.requests if r.path == f"/v1/futures/{run.info_.ready_future_id}"]) == 3
        run.close()

    def test_a_long_wait_for_a_gpu_prints_one_line(
        self,
        client: EndorClient,
        fake: FakeEndor,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Python hides INFO logs by default, so a wait over WAIT_NOTICE_S (a cold GPU takes minutes) also prints one
        line to stderr, unless ``progress=False``."""
        with client.runs.create(BASE):  # ready at once: nothing printed
            pass
        assert capsys.readouterr().err == ""
        monkeypatch.setattr(endor.training, "WAIT_NOTICE_S", 0.0)
        fake.provisioning_polls = 3
        with client.custom_models.create(unique("cold"), base_model=BASE) as run:
            pass
        notice = "waiting for a GPU (starting a cold one can take a few minutes)…"
        assert capsys.readouterr().err.splitlines() == [f"run {run.id}: {notice}"]
        fake.provisioning_polls = 3
        with client.runs.create(BASE, progress=False):
            pass
        assert capsys.readouterr().err == ""
        fake.provisioning_polls = 4
        with client.runs.create(BASE, wait=False) as run, caplog.at_level(logging.INFO, logger="endor"):
            endor.training._wait_ready(run, log_every=0.0)  # many timeouts, each logged
        assert capsys.readouterr().err.splitlines() == [f"run {run.id}: {notice}"]  # once
        assert len([r for r in caplog.records if "still provisioning" in r.getMessage()]) >= 2

    def test_no_wait(self, client: EndorClient, fake: FakeEndor) -> None:
        fake.provisioning_polls = 1
        with client.runs.create(BASE, wait=False) as run:
            assert not run.ready.done() and run.info_.status == "provisioning"
            run.forward(endor.data.rows_to_datums(rows(1))).result()  # ops queue behind provisioning
            run.ready.result()


class TestSavedModels:
    def test_loss_is_the_one_at_save_time(self, client: EndorClient) -> None:
        v0, v1 = unique("v0"), unique("v1")
        with client.runs.create(BASE) as run:
            run.save_checkpoint(v0).result()  # nothing trained yet
            run.forward_backward(endor.data.rows_to_datums(rows(1))).result()  # loss at step 0
            run.optim_step().result()
            run.save_checkpoint(v1).result()  # step 1: the last loss at or before it is step 0's
            run.log({"train/loss": 9.0}, step=5)  # the run trains on
        at0 = [p.value for p in run.metrics(keys=["train/loss"]) if p.step == 0]
        assert client.custom_models.get(v0).loss == at0[0] and client.custom_models.get(v1).loss == at0[0]


class TestEvaluations:
    def test_evaluate_and_list(self, client: EndorClient, fake: FakeEndor) -> None:
        client.datasets.upload("heldout", rows(6))
        name = unique("ev")
        with client.custom_models.create(name, base_model=BASE) as run:
            run.save_checkpoint().result()
        ev = client.evaluations.create(name, "heldout", run_id=run.id).result()  # an inactive model
        assert requests_to(fake, "/v1/evaluations", "POST")[-1].body["model"] == name
        assert ev.status == "completed" and ev.source == "server" and ev.dataset == "heldout"
        assert ev.training_run_id == run.id and ev.model == name
        assert ev.results and 0 <= ev.results["overall"]["accuracy"] <= 1
        base_ev = client.evaluations.create(BASE, "heldout").result()  # the untuned base, as a baseline
        assert base_ev.model == BASE
        inline = client.evaluations.create(name, rows=rows(2)).result()
        assert inline.dataset is None and inline.rows_total == 2
        assert [e.id for e in client.evaluations.list(run_id=run.id)] == [ev.id]
        assert len(client.evaluations.list(model=name)) == 2 and len(client.evaluations.list()) == 3
        assert client.evaluations.get(ev.id).id == ev.id
        with pytest.raises(NotFoundError):
            client.evaluations.create(name, "nope")
        with pytest.raises(NotFoundError) as e:
            client.evaluations.create("not-a-model", "heldout")
        assert e.value.code == "unknown_model"
        with pytest.raises(ValueError, match="one of the two"):
            client.evaluations.create(name)
