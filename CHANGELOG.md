# Changelog

All notable changes to the `endor` package. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
