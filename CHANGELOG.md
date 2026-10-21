# Changelog

All notable changes to the `endor` package. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

0.9.0: projects are continuous learning projects, with a live version and a preview like deployments, and custom
models live outside projects. This release breaks code written for 0.8.0; the table says what to change.

### Breaking changes and how to migrate

| 0.8.0 | 0.9.0 |
|---|---|
| `client.projects.create(name, kind="managed", base_models=["decider-2b"])` | `client.projects.create(name, base_model="decider-2b")`: every project is a continuous learning project, on one base model chosen once. `kind=` and `base_models=` raise a `TypeError` that says this. |
| `kind="custom"` projects, `project.datasets`, `project.runs`, `project.models`, `project.evaluate`, `project.evaluations` | `client.datasets`, `client.runs`, `client.custom_models`, `client.evaluations.create` and `client.evaluations.list`, at the organization's level |
| `model="tickets/v1"` (a saved model), `"tickets/decider-2b"` (a project's base model) | a custom model is called by its name, `model="router-2"`, once promoted (`client.custom_models.promote("router-2")`); a project is `model="tickets"` (its live version) or `"tickets:preview"` (its newest). The `<project>/<name>` form no longer resolves. |
| `run.save_checkpoint("v1")` → `"tickets/v1"`, callable at once | `run.save_checkpoint("router-2")` → `"router-2"`, saved inactive: promote it to call it. `client.custom_models.create("router-2", base_model=...)` opens a run whose `save_checkpoint()` (no name) saves under that name. Each save is a new model; names are unique in the organization with the projects. |
| `project.runs.create()` on the project's first base model | `client.runs.create("decider-2b")` (or `from_model=`): a base model is required |
| `project.add_base_model`, `project.remove_base_model`, `ProjectInfo.kind`, `.base_models`, `.n_datasets`, `.n_runs`, `.n_models`, `.wandb`, `project.update(wandb=...)`, `WandbSettings` | gone. `ProjectInfo` has `base_model`, `live`, `preview`, `auto_promote`, `promoted_at`, `versions`. Runs log to Weights & Biases when the organization has connected it (`runs.create(wandb=False)` turns it off per run). |
| `ModelInfo` (`id`, `kind`, `project`) | `CustomModelInfo` (`name`, `live`, `live_at`, ...) and, for a project's versions, `VersionInfo` |
| `RunInfo.project`, `Run.project`, `DatasetInfo.project`, `Evaluation.project` | gone; `RunInfo.model` / `Run.model` is the custom model name a save uses |
| `WhoAmI.managed_projects` | gone; `limits` has `max_projects_per_org` and `max_live_models_per_org` |
| `NoBaseModelError`, `WrongProjectKindError` | gone (they can't happen any more) |
| `client.predict("tickets/v1", rows)` | `client.predict("router-2", rows)`: a custom model (live or not) or a base model id |
| `SupervisedConfig(project="tickets", model_name="v1")` | `SupervisedConfig(model_name="router-2")`: the recipe saves a custom model, inactive; it scores the untuned base with one server-side prediction |
| CLI `endor models ...`, `datasets list PROJECT`, `runs list PROJECT`, `eval PROJECT MODEL DATASET`, `projects add-base-model` | `endor custom-models ...`, `datasets list`, `runs list`, `eval MODEL DATASET`; `projects versions`, `promote`, `auto-promote` are new |

### Added

- Continuous learning projects like deployments: `project.ref` (`"tickets"`) and `project.preview_ref`
  (`"tickets:preview"`); `project.versions() -> list[VersionInfo]` (newest first, `live`, `preview`, `accuracy`,
  `loss`); `project.promote(version)` (promoting an older version than the preview turns auto-promote off, like a
  rollback); `project.set_auto_promote(on)` (on: the preview goes live at once, and every new version after it; the
  default); `project.download(version, path)`.
- `client.custom_models`: `list(live=None)` (live ones first), `get(name)`, `create(name, base_model, *, run_name=None,
  **run_options) -> Run`, `promote(name, replace=None)` (`replace` swaps out a live model when every slot is taken),
  `demote(name)`, `set_ttl(name, ttl_seconds)`, `download(name, path)`, `delete(name)`. New type `CustomModelInfo`.
- `client.datasets`, `client.runs` (`create(base_model, *, model=None, ...)`, `get`, `list`) and
  `client.evaluations` (`create(model, dataset=None, run_id=None, *, rows=None)`, `get`, `list`) at the
  organization's level.
- Errors: `ModelNotLiveError` (409 `model_not_live`: a decision named an inactive custom model), `LiveModelsFullError`
  (409 `live_models_full`, a `LimitReachedError`, never retried), `ModelIsLiveError` (409 `model_is_live`: deleting a
  live model or giving it a TTL), `VersionNotCallableError` (404 `version_not_callable`: `"<project>:v3"`).
- CLI: `endor projects versions | promote | auto-promote`, `endor custom-models list | info | promote | demote |
  download | set-ttl | delete`.

### Also in this release

- `client.corrections`: tell Endor whether past decisions were right and, if not, the right answers; continuous
  learning projects learn from them. `create(request_id, questions, *, comment=None) -> CorrectionInfo`
  (`POST /v1/corrections`), `batch(items)` (`POST /v1/corrections/batch`: 1 to 1,000 items, all or none),
  `get(request_id)` and `delete(request_id)`, each with an `_async` variant. `request_id` is a `SystemOneResponse` or
  its id. Each question is `{"correct": True}`, `{"correct": False, "answer": ...}` or `{"correct": False}`; an answer
  must be exactly one of the question's options. New types `CorrectionInfo`, `QuestionCorrectionInfo`,
  `QuestionCorrection` and `CorrectionAnswer`.
- `SystemOneResponse.id`: the decision's id (`req_...`, also its `X-Request-Id`). It is None from a server that doesn't
  send one.
- `Learning.drift` (new type `Drift`): per question, whether recent traffic moved away from what the served version
  was checked on. `Training.decisions`: the project's decision count when the run started.
- A continuous learning project's versions are named `v1`, `v2`, ... (the docstrings said `YYYY-MM-DD-N`).

- `endor.metrics.decision_metrics`: when k options tie exactly for the top probability, each counts 1/k toward
  accuracy (so a decision scores the share of its tied top options that are the target's), and a soft target counts
  each of its tied top options as correct. Before, the first option in the dict won the tie, so accuracy depended on
  the options' order. Server-side evaluations use the same rule.
- A row label shaped like a `Datum` target (`{"probs": {...}}` or `{"label": ...}`) raises a ValueError that names
  both shapes, instead of a TypeError from `float()`. A soft label with a non-number value raises ValueError too.
- A `Target` (or a `Datum`'s `target`) given a bare probability map or an option key raises a ValueError that names
  `{"label": ...}` and `{"probs": ...}`, instead of pydantic's "Extra inputs are not permitted".
- `recipes.supervised.train` prints a warning before training for each question with options that have no
  training labels (the model can't learn them) or fewer than 10.

- `project.learning() -> Learning`: a continuous learning project's learning (always on): its `status`, `progress` to
  the next version, what it serves, and the latest `trainings` (kept or discarded, each with its accuracy and loss).
  New types `Learning` and `Training`.
- `project.update(paused=...)` and `ProjectInfo.paused` are gone: learning can't be paused (the API refuses `paused`
  with 422).

Fine-tuning with images, in limited access.

- `endor.Image` objects go in training data too, on base models that read images: rows of `client.datasets.upload`
  and `client.predict`, datums of `run.forward` / `run.forward_backward`, rows of `run.predict`. The SDK uploads each
  image once (`client.images.upload(images) -> list[Image]` does it by hand: declares them with `POST /v1/images`,
  PUTs the files straight to storage, confirms them) and sends references, `{"type": "image", "sha256": ...}`
  (`Image.ref(sha256, media_type=None)`, `image.is_ref`, `image.sha256`, `image.content`, `Image.wrap(dict)`). A
  call carries at most `MAX_IMAGES_PER_CALL` (128) images, so larger batches are split as they are over 1,024 datums.
- Until the organization is approved these calls raise `PermissionDeniedError` (403 `image_training_not_enabled`),
  whose message says how to ask; `client.whoami().image_training` says whether it is. Decisions with images need no
  access.
- `DatasetInfo.n_images`: the images across a dataset's rows.

## [0.8.0] - 2026-10-15

Your prepaid balance from the SDK, and clearer docs on model names and resuming runs.

- `client.balance()` returns the org's prepaid balance in USD (`GET /v1/balance`): credits bought or granted, minus
  usage so far. `endor whoami` shows it too (`balance_usd`).
- Docs: `save_checkpoint` says each model name is saved once (a used name raises `ConflictError` at once; the run is
  unaffected, but an uncaught error leaves the `with` block, which closes the run) and how to name checkpoints in a
  long run; `runs.get` says `run.info_.step` is the step to resume from; `datasets.upload` says how to run a pipeline
  again. The `Run` docstring and the README no longer contradict themselves on run limits (25 open runs; a run left
  open is closed after an hour).

## [0.7.0] - 2026-10-14

Training steps no longer wait for each other's results.

- `endor.Steps(in_flight=2)` keeps up to `in_flight` training steps sent and not yet read: `steps.add(fb, opt)`
  takes a step's futures and returns the results of the steps it waited for (the step before, once two are out),
  oldest first, as `endor.gather` returns them; `steps.wait()` reads the rest (`add_async` and `wait_async` for async
  code). The trainer has the next step queued when it finishes one, instead of waiting for your round trip between
  steps (about 0.7 s a step). A failed step raises from the call that reads it, and a call sent between two `add`
  calls runs after every step added so far.
- `recipes.supervised.train` (and `distill`) sends each step before the last one's results are read:
  `SupervisedConfig.steps_in_flight`, default 2 (1 waits for each step, as before). Results, metrics, the progress
  line and the held-out evaluations are the same; an evaluation reads every step first and runs right after its step.
  A step that fails still raises at that step; the one step sent after it runs before the run is closed.
Predictions for many rows at once, without labels or a dataset.

- `client.predict(model, rows, *, progress=True, timeout=None)` (and `predict_async`) returns the answers to every
  question of every row, in the rows' order: one `{question name: NoulAnswer | ChoiceAnswer | ScoreAnswer}` per row,
  as `system_one(...).answers`. It sends the rows without their labels as one server-side evaluation in the model's
  project (`POST /v1/projects/{project}/evaluations` with `rows`), waits for it, printing a progress line to stderr
  every 10 seconds while it runs (`evaluation evl_...: 1,200/10,000 rows (12%) · ~2 min left`), and fetches its
  results file from storage (HTTPS only, without the API key). At most 50,000 rows and 32 MB per call; billed as
  decisions. `TimeoutError` after `timeout` seconds (the evaluation keeps running), `OperationFailedError` when it
  fails.
- `project.evaluate(model, dataset=None, run_id=None, *, rows=None)` takes rows inline instead of a dataset's name
  (one of the two). A row with labels is scored on its labeled questions; a row without labels is answered on every
  question and not scored (its answers are in `results["rows_url"]`).
- `Evaluation` has `rows_total`, `rows_done` and `started_at`, the API's progress of a server-side evaluation.
- Every request is timed end to end, from sending it to reading the response, and that time is sent with the next
  request, once, as `X-Endor-Client-Latency: <request id>=<ms>`, so the dashboard can show the latency your app sees
  next to Endor's own. Nothing waits for it and it never raises. `EndorClient(report_latency=False)` or
  `ENDOR_REPORT_LATENCY=0` turns it off.

## [0.6.1] - 2026-10-13

- User reports need an API key, as the API now requires: `endor.user_report` without `api_key` or `ENDOR_API_KEY`
  raises `EndorError` ("no API key", as `EndorClient`) before sending, and a key the API refuses is an
  `AuthenticationError`. `endor user-report` needs `ENDOR_API_KEY` like every other command.

## [0.6.0] - 2026-10-12

User reports to the Endor team, for people and their coding agents.

- `endor.user_report(message, *, kind="bug", request_id=None, context=None, api_key=None, base_url=None, timeout=None)`
  and `client.user_report(message, *, kind="bug", request_id=None, context=None)` (and their `_async` variants) send
  `POST /v1/user_reports`: `kind` is `bug`, `confusing` or `works`. They return a `UserReportReceipt` (`id`, `message`).
  `endor.user_report` needs no client and no key (anonymous; with `ENDOR_API_KEY` the team sees your org). By default a
  report is about the latest failed request (the client's, or the process's for `endor.user_report`) and carries the SDK
  and Python versions, the platform and that failure's status, code and endpoint; never a key, a request body or a
  decision's content. A 429 is a `RateLimitError` and is never retried.
- Every error's `str()` ends with one line saying how to report it, with the request id when there is one
  (`endor.errors.report_hint`); `APIError.message` and the other attributes are unchanged, and `repr()` leaves the
  line out.
- CLI: `endor user-report "message" [--kind bug|confusing|works] [--request-id REQ]`, with or without `ENDOR_API_KEY`;
  other commands' errors end with that command.

## [0.5.1] - 2026-10-10

- Docs: managed projects are coming soon. `projects.create(kind="managed")` is refused by the API (422
  `invalid_input`) until `client.whoami().managed_projects` is true; the docstrings and README now say so.

## [0.5.0] - 2026-10-09

Projects have several base models. The API refuses `base_model` on `POST /v1/projects` (422 `invalid_input`), so
`projects.create` and `get_or_create` failed on 0.4.x; update to 0.5.0.

- `projects.create(name, description=None, *, kind="custom", base_models=None)` and `get_or_create(...)` take
  `base_models`, a list of base model ids (`["decider-2b"]`). Each answers at once as `"<project>/<base id>"`, and
  `"<project>"` answers with the first. A managed project needs at least one. `base_model=` raises `TypeError`
  saying to pass `base_models=[...]`. `get_or_create` adds the `base_models` an existing project doesn't have yet
  (it removes none, and keeps the description); it raises `ValueError` only for another `kind`.
- `project.add_base_model(base_model)` and `project.remove_base_model(base_model)` (`POST` and
  `DELETE /v1/projects/{project}/base_models`); the CLI has `endor projects add-base-model NAME BASE` and
  `remove-base-model NAME BASE`, and `projects create` takes `--base-model` any number of times (or none).
- `ProjectInfo.base_models` (in the order they were added) replaces `ProjectInfo.base_model`.
- Base models are called by their ids: `"<project>/decider-2b"` replaces `"<project>/base"`, which the API now
  answers with 404 `unknown_model`. `project.evaluate("decider-2b", ...)`, `project.models.get("decider-2b")`; the
  evaluation records the full id (`"<project>/decider-2b"`). `ModelInfo.kind` is `"base"` or `"saved"`, and a base
  model's `name` is its id. Saved models can't be named `base` or after a base model.
- `runs.create()` trains on the project's first base model, or on `base_model=`; a base the project doesn't have yet
  is added to it (it used to be refused). The supervised recipe adds its `base_model` to the project the same way
  and scores the base as `"<project>/<base_model>"`.
- Fields the API returns: `ModelInfo.accuracy`, `WhoAmI.managed_projects`, `UsageRow.project_deleted`,
  `BaseModelInfo.provider`, `leaderboard_rank` and `max_question_tokens`.
- `WandbSettings` is `enabled` and `url` (the W&B project page), as the API returns it; the W&B project is
  `endor-<project name>`. `project.update(wandb={"enabled": True})` sends only `enabled` (`entity` and `project`
  are gone: the API never took them).

## [0.4.2] - 2026-10-08

- Images in decisions: `endor.Image.from_path(path)`, `Image.from_bytes(data, media_type=None)` and
  `Image.from_pil(image, format="PNG")` (Pillow, not installed by the SDK) build the state object
  `{"type": "image", "media_type": ..., "data": <base64>}`; put it anywhere in a state on a base model that reads
  images. `BaseModelInfo.modalities` says which do (`["text", "image"]`); the others answer
  `422 unsupported_modality` (`UnprocessableEntityError`).

## [0.4.1] - 2026-10-07

- Decisions wait out a cold start. A base model that has been idle takes a few minutes to start, and the API waits up
  to 300 s before it answers `503 warming_up` with `Retry-After`. `system_one` and `system_one_async` now wait up to
  330 s per request (other calls keep 30 s) and retry within 660 s instead of 60 s, so the first decision is answered
  instead of timing out and being sent (and billed) twice. A `timeout` or `retry` you give the client or the call
  still wins.

## [0.4.0] - 2026-10-07

- Training progress: `runs.create(total_steps=N)` and `run.set_total_steps(N)` tell Endor how many optimizer steps
  you plan, so the dashboard shows a progress bar and an ETA. `RunInfo` has `total_steps`, `seconds_per_step` (the
  median over the last 20 steps), `progress` (0 to 1) and `eta_seconds`. The supervised recipe (and distill) sets
  `total_steps` from the data size and epochs and prints a progress line (`SupervisedConfig(progress=False)` turns
  it off).
- Weights & Biases, logged by Endor's servers through the organization's W&B connection (dashboard: Settings >
  Integrations): `project.update(wandb={"enabled": True, "entity": ..., "project": ...})` turns it on for a
  project's new runs, `runs.create(wandb=True|False)` overrides it per run. `ProjectInfo.wandb` (`WandbSettings`:
  `enabled`, `entity`, `project`); `RunInfo.wandb` and `wandb_url`.
- `RunInfo.ready_at` (the run's GPU was ready) and `closed_at`.
- `exclude_from_training`: `system_one(..., exclude_from_training=True)` (or `EndorClient(exclude_from_training=True)`
  for every call) keeps a decision out of continuous learning, for benchmarks and evaluations; it is still answered,
  billed and logged. Sent only when on, so other requests are exactly TypeSafe's.

## [0.3.0] - 2026-10-07

Breaking: there is no live model any more, as in the API.

- Removed `project.set_live`, `ProjectInfo.live_model` and `ModelInfo.live`. Call each saved model by its id,
  `"<project>/<name>"`. The bare `"<project>"` answers with a managed project's newest version (the base until one
  exists) and with a custom project's base model, like `"<project>/base"`.
- `client.models.list()`: the `"<project>"` entry's `kind` is `project` (was `live`).
- `ModelInfo.loss`: the training loss when the model was saved (its run's `train/loss` at the model's step, or the
  last one before it).
- `WhoAmI.limits` has `decisions_per_minute` (60 by default, shared by all keys of the organization) instead of
  `decisions_per_second` and `decisions_burst`.

## [0.2.3] - 2026-10-07

- `Score` explains its `criteria`: the levels of one scale, lowest first, not a list of things to check.

## [0.2.2] - 2026-10-07

- A custom project may be created without `base_model`: its first `runs.create(base_model=...)` sets it, for good.
  Managed projects still need it at creation.

## [0.2.1] - 2026-10-07

- `WhoAmI` no longer has `key_prefix`: the API stores only a hash of each key.
- `WhoAmI.limits` keeps whole numbers as integers.

## [0.2.0] - 2026-10-07

Breaking: projects have a kind, and keep warm and auto-promote are gone, as in the API.

- Project kinds, set at creation and never changed: `projects.create("tickets", base_model=..., kind="custom")`
  (the default; you train with the SDK) or `kind="managed"` (Endor trains new versions from the project's
  decisions; its decisions are billed at the continuous-learning price, 50% more). `base_model` is now required by
  `create` and `get_or_create`, and can't change. `get_or_create` raises `ValueError` when the existing project has
  another kind or base model.
- `ProjectInfo` gains `kind` and `paused` (managed only); `project.update(description=..., paused=...)`. Removed:
  `continuous_learning` (and `ContinuousLearning`), `auto_promote`, `base_keep_warm`, and `update(base_model=...)`.
- `WrongProjectKindError` (409 `wrong_project_kind`, a `ConflictError`): a managed project refuses datasets, runs,
  evaluations and `set_live`; a custom project refuses `paused`.
- `runs.create()` may leave out `base_model`: runs train on the project's base model (another base is a 422).
- Removed `project.models.set_keep_warm`, `ModelInfo.keep_warm`, `ModelInfo.source` and `RunInfo.source`.
- Model names may start with a digit; only `base` is reserved.
- CLI: `endor projects create NAME --base-model BASE [--kind custom|managed]`.

## [0.1.1] - 2026-10-07

- `project.set_live(model)`: make a saved model (or `"base"`) the live model, what `model="<project>"` answers with,
  through the new `POST /v1/projects/{project}/live`. It turns `auto_promote` off, as Make live on the dashboard does.
- `client.models.list()` lists the names you can pass as `model`: `"<project>"` and `"<project>/base"` for each
  project with a base model, then `"<project>/<name>"`. `ModelMetadata.kind` is `live`, `base` or `model`.
- `client.base_models()` reads the new `GET /v1/base_models` (falling back to `/v1/models` on an older API).
  `BaseModelInfo` gains `params`, `hf_revision`, `contract_version`, `max_rank` and
  `price_per_mtok_decide_continuous_learning`.
- `SupervisedConfig.base_model` defaults to `decider-2b` (was `pplx-decider-v1.1-27b`).
- Score labels also take the option id: `"2"` as well as `2`.
- Docs: `ENDOR_BASE_URL` for staging, continuous learning marked as coming soon, GPU start-up usually under a minute.

## [0.1.0] - 2026-10-06

First release. Matches the current API (org API keys, GPU-hour billing, balances).

- Every decision names a project: `model` is `"<project>"` (its live model), `"<project>/base"` (its base model) or
  `"<project>/<name>"`. A bare base model id raises `ModelRequiresProjectError` (422 `model_requires_project`), a
  project without a base model `NoBaseModelError` (409 `no_base_model`). `project.evaluate("base", ...)` and bare
  names resolve in the project.
- No default model any more (it was `pplx-decider-v1.1-27b`): `ENDOR_DEFAULT_MODEL` is optional, and
  `system_one` without a model raises `EndorError` before sending. `endor.client.DEFAULT_MODEL` is gone.
- Projects have a base model: `projects.create(..., base_model=...)` (also `get_or_create`), and
  `project.update(base_model=..., auto_promote=..., base_keep_warm=...)`. `ProjectInfo` gains `base_model`,
  `live_model`, `auto_promote` and `base_keep_warm`. `project.models.set_keep_warm("base", on)` sets
  `base_keep_warm`.
- `ModelInfo.source` (`base`, `sdk` or `continuous`) and `ModelInfo.live`; `project.models.list()` starts with the
  base model. `RunInfo.source` (`sdk` or `continuous`).
- `WhoAmI.limits`: the org's limits by name.
- `LimitReachedError` (409 `limit_reached`, a `ConflictError`, never retried) for every count limit: 4 open runs
  (was 429 `quota_exceeded`), 7 projects, 50 models per project, 3 kept-warm models per project (was 409
  `conflict`, per org and base).
- `save_checkpoint` checks the model name locally, as the API does: it starts with a letter and is not `base`.
- The supervised recipe creates its project with `base_model` and scores the baseline as `"<project>/base"`.
- `endor projects create NAME --base-model BASE`.
- `InsufficientBalanceError` (402 `insufficient_balance`), never retried; `quota_exceeded` is no longer retried.
- `whoami()` returns `org_id`; `user_id` is None for an org API key.
- `usage()` rows: `kind` is `decide` or `train`, with `model`, `input_tokens` and `gpu_seconds`; `tokens` is gone.
  Decide rows also carry `continuous_learning`, `price_per_mtok`, `base_cost_usd` and `continuous_learning_cost_usd`.
  The CLI's `usage --csv` columns follow.
- New fields: `BaseModelInfo.price_per_gpu_hour` (and `contract`, `hf_repo`, `trainer_gpu`; `price_per_mtok_train`
  is gone), `ModelInfo.parent_model` and `contract`, `RunInfo.contract`. Responses are parsed tolerantly.
- Runs are closed reliably: `runs.create(wait=True)` closes the run if waiting is interrupted, `Run` is an async
  context manager too, a `with` block that fails still closes the run, and `supervised.train` closes on errors.
- `forward` and `forward_backward` validate every datum locally before sending any chunk; if the server refuses a
  later chunk, the accepted ones are cancelled.
- `runs.create(from_model=...)` inherits the saved model's LoRA settings: only LoRA arguments you pass are sent.
- Base models are now `pplx-decider-v1.1-27b` (the default), `gev-26b`, `jev-9b`, `decider-2b` and
  `gliner2.5-decide`.
- `project.models.set_keep_warm(model, on)`; `ModelInfo.keep_warm`.
- `project.models.download(model, path, include_optimizer=False)` writes the model's files into a folder from
  short-lived per-file links, verifying size and SHA-256 (`DownloadError` otherwise), and returns `DownloadedFile`s.
  It replaces `archive_url` and `ArchiveInfo` (the API's `archive_url` route is gone). The CLI's
  `endor models download MODEL [-o DIR] [--include-optimizer]` writes a folder instead of a tar.
- Continuous learning settings: `projects.create(..., continuous_learning=...)`, `project.update(...)`,
  `ProjectInfo.continuous_learning`.
- `run.log()` without a step logs at the run's current step on the server.
- List calls (`projects.list`, `datasets.list`, `runs.list`, `models.list`, `evaluations`) page through everything.
- `EndorClient(timeout=...)` is used as given; it is no longer raised to 30 s.
- Identification: `User-Agent` is now `endor-python/<version> (python <version>; <os>; <cpu>[; cli])`, next to the
  existing `X-Endor-SDK`, `X-Endor-SDK-Version`, `X-Endor-Runtime` and `X-Endor-SDK-Interface` headers.
- `usage()` takes datetimes without a timezone as UTC.


- `EndorClient`: `system_one` decisions (sync and async), the model catalog, `whoami` and `usage`.
- Projects with datasets, runs, models and evaluations.
- Runs: `forward`, `forward_backward`, `optim_step`, `save_checkpoint`, metrics and client-side evaluations, with
  strict sequence numbers and safe retries.
- Futures with long polling, batched waiting (`endor.gather`) and cancellation.
- `endor.data` for loading, converting, splitting and batching decision rows; `endor.metrics` for decision metrics.
- Recipes: `supervised.train` and a teacher `distill`.
- The `endor` command line.
- SDK identification headers on every request (see "What the SDK sends" in the README).
