# Changelog

All notable changes to the `endor` package. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
