# What the SDK sends

Every request the SDK sends carries a few headers that say where it came from. They let you trace a request in your
own logs back to the SDK call that made it, and they let Endor support you and see which SDK versions are in use.

| Header | Example | On |
|---|---|---|
| `Authorization` | `Bearer edk_…` | every request |
| `Accept` | `application/json` | every request |
| `Content-Type` | `application/json` | requests with a body |
| `Content-Encoding` | `gzip` | bodies over 32 KB (on by default; `EndorClient(gzip=False)` turns it off) |
| `User-Agent` | `endor-python/0.1.0 (python 3.12.1; darwin; arm64)`, with `; cli` added for the `endor` command | every request |
| `X-Endor-SDK` | `endor-python` | every request |
| `X-Endor-SDK-Version` | `0.1.0` | every request |
| `X-Endor-Runtime` | `python/3.12.1 (darwin; arm64)` | every request |
| `X-Endor-SDK-Interface` | `python` or `cli` | every request |
| `X-Endor-SDK-Method` | `run.forward_backward` | every request: the public SDK method that caused it (list below) |
| `X-Endor-SDK-Recipe` | `supervised` or `distill` | requests made inside a recipe |
| `X-Endor-Client-Request-Id` | a UUID4 | every request; the same value on every retry of one logical call |
| `X-Endor-Retry-Count` | `1`, `2`, … | retries only |
| `Idempotency-Key` | a UUID4 | resource-creating POSTs (see below); the same value on every retry |

The SDK version, the Python version, the operating system and the CPU architecture are all that is sent about your
machine.

Headers you pass through `EndorClient(headers=...)` or `extra_headers=` are sent too. The identity headers above
(`User-Agent`, `X-Endor-SDK*`, `X-Endor-Runtime`), `Authorization` and `Accept` cannot be overridden.

**What Endor keeps.** For every request, the API stores the request id, your org, the calling key, the route, the
status and the latency, plus a `client` record with one field per header: `user_agent`, `sdk`, `sdk_version`,
`runtime`, `interface`, `method`, `recipe`, `client_request_id` and `retry_count` (each cut to 200 characters). It
also keeps the request body, and the response body when it is JSON and at most 256 KB, in object storage (S3). The
secret API key itself is never stored.

**In the body.** `project.runs.create` also sends the run's settings and, unless you create the client with
`capture=False`, the current git commit of your working directory (`<sha>` or `<sha>+dirty`; never file contents),
so the dashboard can show how a model was made.

## `X-Endor-SDK-Method` values

| Value | Request |
|---|---|
| `client.system_one` | `POST /v1/systemone` |
| `client.models.list` | `GET /v1/models` |
| `client.base_models` | `GET /v1/models` |
| `client.whoami` | `GET /v1/whoami` |
| `client.usage` | `GET /v1/usage` |
| `projects.create`, `projects.get`, `projects.list` | `/v1/projects…` |
| `project.info`, `project.delete` | `/v1/projects/{project}` |
| `project.evaluate` | `POST /v1/projects/{project}/evaluations`, then the future and `GET /v1/evaluations/{id}` |
| `project.evaluations` | `GET /v1/projects/{project}/evaluations` |
| `project.datasets.upload`, `.list`, `.get`, `.rows`, `.delete` | `/v1/projects/{project}/datasets…` |
| `project.runs.create`, `.get`, `.list` | `/v1/projects/{project}/runs`, `/v1/runs/{id}` |
| `project.models.list`, `.get`, `.set_ttl`, `.archive_url`, `.delete` | `/v1/projects/{project}/models…` |
| `run.ready` | polling the run's provisioning future |
| `run.forward`, `run.forward_backward`, `run.optim_step`, `run.save_checkpoint` | the op's POST, the polls of its future, and the cancels of a batch refused part-way |
| `run.close`, `run.info`, `run.log`, `run.metrics`, `run.log_eval` | `/v1/runs/{id}/…` |
| `future.result`, `future.cancel`, `gather` | `/v1/futures/…` for futures polled outside a run method |

Polls of a future (`GET /v1/futures/{id}` and `POST /v1/futures/retrieve`) carry the method that created the future,
so a long training step shows up under `run.forward_backward` from submit to result.

## `Idempotency-Key`

Sent on the POSTs that create a resource, so a request the SDK retries after a lost response never creates a
duplicate: `projects.create`, `project.datasets.upload`, `project.runs.create`, `project.evaluate` and
`run.log_eval`. The compute and save calls of a run do not need it: they carry a sequence number instead, and a
retried call reuses its number and body.

## Logging

The SDK logs one INFO line per response on the `endor` logger (method, path, status, latency and the server's
`x-request-id`), and the full headers at DEBUG with credentials redacted. Set `ENDOR_LOG_LEVEL=info` or configure
the logger yourself. Request and response bodies are never logged by the SDK.
