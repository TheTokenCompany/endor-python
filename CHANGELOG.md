# Changelog

All notable changes to the `endor` package. The format follows [Keep a Changelog](https://keepachangelog.com/) and
the version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- SDK identification headers on every request (see `docs/REQUEST_HEADERS.md`).
