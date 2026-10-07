# Changelog

All notable changes to the `endor` package. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

Matches the current API (org API keys, GPU-hour billing, balances).

- `InsufficientBalanceError` (402 `insufficient_balance`), never retried; `quota_exceeded` is no longer retried.
- `whoami()` returns `org_id`; `user_id` is None for an org API key.
- `usage()` rows: `kind` is `decide` or `train`, with `model`, `input_tokens` and `gpu_seconds`; `tokens` is gone.
  The CLI's `usage --csv` columns follow.
- New fields: `BaseModelInfo.price_per_gpu_hour` (and `contract`, `hf_repo`, `trainer_gpu`; `price_per_mtok_train`
  is gone), `ModelInfo.parent_model` and `contract`, `RunInfo.contract`. Responses are parsed tolerantly.
- Runs are closed reliably: `runs.create(wait=True)` closes the run if waiting is interrupted, `Run` is an async
  context manager too, a `with` block that fails still closes the run, and `supervised.train` closes on errors.
- `forward` and `forward_backward` validate every datum locally before sending any chunk; if the server refuses a
  later chunk, the accepted ones are cancelled.
- `runs.create(from_model=...)` inherits the saved model's LoRA settings: only LoRA arguments you pass are sent.
- `run.log()` without a step logs at the run's current step on the server.
- List calls (`projects.list`, `datasets.list`, `runs.list`, `models.list`, `evaluations`) page through everything.
- `EndorClient(timeout=...)` is used as given; it is no longer raised to 30 s.
- Identification: `User-Agent` is now `endor-python/<version> (python <version>; <os>; <cpu>[; cli])`, next to the
  existing `X-Endor-SDK`, `X-Endor-SDK-Version`, `X-Endor-Runtime` and `X-Endor-SDK-Interface` headers.
- `usage()` takes datetimes without a timezone as UTC.

## [0.1.0] - 2026-10-06

First release.

- `EndorClient`: `system_one` decisions (sync and async), the model catalog, `whoami` and `usage`.
- Projects with datasets, runs, models and evaluations.
- Runs: `forward`, `forward_backward`, `optim_step`, `save_checkpoint`, metrics and client-side evaluations, with
  strict sequence numbers and safe retries.
- Futures with long polling, batched waiting (`endor.gather`) and cancellation.
- `endor.data` for loading, converting, splitting and batching decision rows; `endor.metrics` for decision metrics.
- Recipes: `supervised.train` and a teacher `distill`.
- The `endor` command line.
- SDK identification headers on every request (see "What the SDK sends" in the README).
