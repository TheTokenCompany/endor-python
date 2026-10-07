<div align="center">

# Endor Python SDK

Decisions with System One models, and fine-tuning on your own data.

[![CI](https://github.com/TheTokenCompany/endor-python/actions/workflows/ci.yml/badge.svg)](https://github.com/TheTokenCompany/endor-python/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/endor)](https://pypi.org/project/endor/)
[![Python versions](https://img.shields.io/pypi/pyversions/endor)](https://pypi.org/project/endor/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TheTokenCompany/endor-python/blob/main/LICENSE)

</div>

A **decision model** takes a `state` (text or JSON) and one or more named questions, and answers each with a
probability for every option in a single forward pass. No text is generated, so there is no sampling and no
temperature. Three question types cover most decisions:

| Type | Answer |
|---|---|
| `Noul` | yes or no, as `P(true)` |
| `Choice` | one of up to 255 named options, with a probability for each |
| `Score` | one of 2 to 10 ordered levels, with a probability for each and the expected level |

Endor serves these models and lets you fine-tune them on your own labeled data, with a training loop you drive from
Python while Endor runs the GPUs.

## Install

```bash
pip install endor
export ENDOR_API_KEY=edk_...
```

Python 3.10 and up. Dependencies: `httpx` and `pydantic`.

The SDK calls production (`https://api.endor.thetokencompany.com`). For another environment, such as staging, set
`ENDOR_BASE_URL=https://staging.api.endor.thetokencompany.com` or pass `EndorClient(base_url=...)`.

API keys belong to an organization (`edk_...`, created in the dashboard); everything you create belongs to that org.

## Decisions

Every decision goes through a project, which records it (for continuous learning, coming soon). Create one
with a base model and it answers at once:

```python
import endor
from endor import Choice, Noul, Score

client = endor.EndorClient()
client.projects.create("tickets", base_model="decider-2b")   # once

res = client.system_one(
    {"subject": "Charged twice", "body": "I was billed two times for my March invoice."},
    {
        "department": Choice(instructions="Which team should handle `body`?",
                             criteria={"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": None}),
        "urgent": Noul(instructions="Does the customer need an answer today?"),
        "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"]),
    },
    model="tickets",             # the project's live model; "tickets/base" or "tickets/v1" for a specific one
)

res.choices["department"].choice          # "billing"
res.choices["department"].probabilities   # {"billing": 0.91, "technical": 0.07, "sales": 0.02}
res.nouls["urgent"].noul                  # 0.83
res.scores["frustration"].score           # 1.4 (expected level), plus .probabilities and .legend
res.model, res.usage.input_tokens         # the model that answered, e.g. "tickets/base"
```

| `model` | The model that answers |
|---|---|
| `"tickets"` | the project's live model: its base model, until you make a saved model live with `project.set_live("v1")` (or Make live on the dashboard's Models tab) |
| `"tickets/base"` | the project's base model, without an adapter |
| `"tickets/v1"` | one saved model |

A bare base model id such as `"decider-2b"` raises `ModelRequiresProjectError` (422 `model_requires_project`); a
project without a base model raises `NoBaseModelError` (409 `no_base_model`). There is no default model: pass
`model=` or set one with `EndorClient(model="tickets")` or `ENDOR_DEFAULT_MODEL`, else `system_one` raises
`EndorError` before sending anything.

Questions can also be plain dicts in the same shape. Every question is answered independently against the same
state. Refer to parts of the state with backticked paths, as in ``"Is `body` angry?"``.

Decisions are TypeSafe's API (`POST /v1/systemone`). The SDK has its own `Noul`, `Choice`, `Score`, answer and
`SystemOneResponse` types, which send and read exactly TypeSafe's wire format, so it doesn't depend on
`typesafe-sdk`; an unmodified TypeSafe client also works against Endor by changing `base_url` and `model`.

**Typed responses.** Subclass `SystemOneResponse` to get one attribute per question:

```python
from endor import ChoiceAnswer, NoulAnswer, SystemOneResponse

class Triage(SystemOneResponse):
    department: ChoiceAnswer
    urgent: NoulAnswer

triage = client.system_one(state, questions, model="tickets", response_model=Triage)
triage.department.choice, triage.urgent.noul
```

**The catalog.** `client.models.list()` returns every name you can pass as `model` (`GET /v1/models`):
`"<project>"` and `"<project>/base"` for each project with a base model, then `"<project>/<name>"` for each saved
model (`.kind` is `live`, `base` or `model`). `client.base_models()` returns the base models (`GET /v1/base_models`)
with their option limits, `hf_repo`, `contract` and prices (`price_per_mtok_decide`, `price_per_gpu_hour`). Call a
base model through a project: `"<project>/base"`.

**Async.** `await client.system_one_async(...)` and `await client.models.list_async()`; `async with EndorClient()`
closes the connections.

## Projects

Everything you create lives in a **project**, keyed by its name. A project has a base model and holds datasets,
runs, the models those runs save (named `<project>/<name>`), and evaluations.

```python
project = client.projects.get_or_create("tickets", base_model="decider-2b")
project.datasets.upload("train", endor.data.load_rows("train.jsonl"))   # rows: see "Data format" below
project.datasets.upload("heldout", endor.data.load_rows("heldout.jsonl"))
baseline = project.evaluate("base", "heldout").result()   # names resolve in the project: "base" is "tickets/base"
```

Without `base_model`, the project's first run sets it. `project.info_` has `base_model`, `live_model` (what
`"tickets"` serves), `auto_promote` and `base_keep_warm`; change them with
`project.update(base_model=..., auto_promote=..., base_keep_warm=...)`. `project.set_live("v1")` makes a saved model
the live model (`"base"` goes back to the base model) and turns `auto_promote` off. A different base model makes `"tickets"`
serve the new base again; saved models stay in the project. `project.models.list()` starts with the base model
(`name == "base"`, `source == "base"`); each model has `source` (`base`, `sdk` or `continuous`) and `live`.

**Continuous learning** (coming soon). A project setting for Endor to keep fine-tuning a model on the project's
decisions, from its base model. The setting exists today; the training pipeline behind it is not available yet.

```python
project = client.projects.create("tickets", base_model="decider-2b", continuous_learning={"enabled": True})
project.update(continuous_learning={"enabled": False})            # pause it
project.info_.continuous_learning   # ContinuousLearning(enabled, base_model, model): model is the newest one
```

Its versions will be named `YYYY-MM-DD-N` (`source == "continuous"`). With `auto_promote` on (the default), each new
version becomes the live model, so `"tickets"` follows it. `auto_promote` never applies to models saved from the SDK;
`set_live` (or Make live on the dashboard) turns it off.

**Keep a model warm.** `project.models.set_keep_warm("v1")` keeps a model loaded on the decision servers, so even
its first request answers without a load time (free; `set_keep_warm("v1", False)` releases it). At most 3 models can
be kept warm in a project, the base model included while its keep warm is on (the default for a new project); a
fourth raises `LimitReachedError` with `param == "keep_warm"`. The live model is always loaded and doesn't count.

**Model names** you save start with a letter (`a-z`), then lowercase letters, digits, `.`, `_` or `-`, up to 63
characters, and can't be `base`. Names starting with a digit are continuous learning's. `save_checkpoint` raises
`ValueError` for any other name before sending it.

A dataset row is a decision request with labels:

```json
{"state": {"body": "I was charged twice"},
 "questions": {"department": {"type": "choice", "criteria": {"billing": null, "technical": null}},
               "urgent": {"type": "noul", "instructions": "Does this need an answer today?"}},
 "labels": {"department": "billing", "urgent": true}}
```

Labels can be hard (`"billing"`, `true`, a level index) or soft (a distribution over the options).

### Data format

A row has `state` (text or JSON), `questions` (named `Noul`, `Choice` or `Score` objects, as `system_one` takes),
and optionally `labels` (question name → label), `id` and `weight` (default 1). Up to 50,000 rows per dataset.
Each labeled question becomes one training example; unlabeled questions are skipped.

Every option has an id, used in labels, training targets and answer probabilities: `"false"`/`"true"` for a noul,
the `criteria` keys for a choice, `"0"`…`"n-1"` (the level index) for a score.

| Type | Hard label | Soft label (every option, non-negative, summing to 1 ± 0.02) |
|---|---|---|
| `noul` | `true` / `false` | `0.8` (P(true)) or `{"true": 0.8, "false": 0.2}` |
| `choice` | `"billing"` | `{"billing": 0.7, "technical": 0.2, "sales": 0.1}` |
| `score` | `2` (level index) or `"2"` (its option id) | `[0.1, 0.6, 0.3]` or `{"0": 0.1, "1": 0.6, "2": 0.3}` |

`endor.data.load_rows(path)` reads `.jsonl` or a `.json` array and also accepts single-question rows,
`{"state", "question", "expected"}` or `{"state", "question", "label"?, "target"?}`, each becoming one question named
`decision`. `endor.data.save_rows(path, rows)` writes `.jsonl`.

## Fine-tune in a few lines

```python
from endor.recipes import SupervisedConfig, supervised

cfg = SupervisedConfig(project="tickets", model_name="v1")   # base_model="decider-2b" by default
result = supervised.train(cfg, endor.data.load_rows("train.jsonl"))

result.base_metrics["accuracy"], result.final_metrics["accuracy"]   # held-out: "tickets/base", then yours
client.system_one(state, questions, model=result.model)             # "tickets/v1"
```

The recipe holds out 10% of the rows, scores the untuned base model (`"<project>/base"`; the project must have
`base_model` as its base, and one without a base gets it) on them, trains one epoch (learning rate `1e-4`
with warmup then linear decay, batches of 16), scores the held-out rows during and after training, and saves the
final model. Every number appears on the run's dashboard page. `SupervisedConfig` has the knobs: base model, rank,
learning rate and schedule, batch size, epochs, loss, how often to evaluate.

## Training costs and limits

- **Billing.** Training is billed per GPU-hour, one price for every GPU type, from when a run's GPU is requested
  (start-up and model loading count) until it is released. Decisions are billed per 1M input tokens, per base model.
  `client.base_models()` shows the prices; `client.usage(starting_on, ending_before)` returns hourly `decide` rows (input tokens)
  and `train` rows (GPU-seconds) with their cost.
- **Idle timeout.** After 15 minutes without calls, a run saves its state and releases its GPU (status `idle`).
  The next call restarts it on a new GPU, which takes a little longer (usually under a minute).
- **4 open runs per org.** Every run that isn't closed counts, idle ones included. A fifth `runs.create` raises
  `LimitReachedError` (409, code `limit_reached`), whose message lists the open runs; close one (`run.close()` or
  `endor runs close RUN_ID`). It is not retried.
- **Other limits.** 7 projects per org, 20 API keys, 50 models per project and 3 kept-warm models per project also
  raise `LimitReachedError` once reached. `client.whoami().limits` lists every limit with its value.
- **Close your runs.** Use `with project.runs.create(...) as run:` (or `async with`): the run is closed when the
  block ends, also on errors. `runs.create(wait=True)` closes the run if you interrupt it while it provisions, and the
  recipes close their run whatever happens. `run.close()` refuses later calls, lets calls already accepted finish
  (status `closing`), then releases the GPU (status `closed`); it returns at once, or waits with `close(wait=True)`.
- **Balance.** When your org's balance is used up (a new org starts with none), decisions, new runs, training calls
  and server-side evaluations raise `InsufficientBalanceError` (402, code `insufficient_balance`) until you add
  credit in the dashboard under Billing. A call the server already accepted still finishes.

## Or drive the loop yourself

```python
rows = endor.data.load_rows("train.jsonl")
train, heldout = endor.data.split(rows, holdout=0.1)
train, heldout = endor.data.rows_to_datums(train), endor.data.rows_to_datums(heldout)

with project.runs.create(base_model="decider-2b", rank=16) as run:
    for batch in endor.data.batches(train, 16):
        fb = run.forward_backward(batch)            # gradients accumulate on the trainer
        opt = run.optim_step(learning_rate=1e-4)    # AdamW step, then zero gradients
        out, step = endor.gather(fb, opt)           # submit both, then wait once
    scored = run.forward(heldout).result()          # option probabilities with the current adapter
    metrics = endor.metrics.decision_metrics(scored.probabilities, [d.target for d in heldout])
    run.log({"heldout/accuracy": metrics["accuracy"]})   # at the run's current step
    model = run.save_checkpoint("v2", include_optimizer=True).result()   # "tickets/v2", usable at once

resumed = project.runs.create(from_model="v2", include_optimizer=True)   # continue exactly where v2 stopped
```

A run created with `from_model` inherits that model's base model and LoRA settings (rank, alpha, which layers are
trained); the SDK sends only the settings you pass explicitly, and one that conflicts with the model is a 422. A
fresh run defaults to rank 16, alpha 32, attention and MLP layers, no readout.

- A `Datum` is one state, one question, a `Target` (a hard `label` or soft `probs`) and a weight. Each labeled
  question of a row becomes one datum.
- `forward`, `forward_backward`, `optim_step` and `save_checkpoint` return futures. Call `.result()`, `await` them,
  or wait for several with `endor.gather(...)`. Each has an `_async` variant that submits without blocking. A future
  can stay pending while a GPU starts (usually under a minute); `.result()` keeps waiting.
- Gradients are the weighted mean over every datum since the last `optim_step`, so splitting a batch across calls
  changes nothing. `optim_step` with nothing accumulated fails with `no_gradients`. Batches over 1,024 datums are
  split for you; every datum is checked locally before any chunk is sent, so a bad datum never leaves half a batch
  applied.
- Losses: `cross_entropy` (hard or soft targets) and `brier`.
- `optim_step` takes `AdamParams` (betas, epsilon, weight decay, gradient clipping) and the learning rate, which you
  pass every step: the schedule is yours.
- `run.forward` scores data with the current adapter without saving a model. `project.evaluate(model, dataset)`
  scores a saved model on a stored dataset server-side.
- `run.close()` releases the trainer (a `with` block does it for you). Unsaved progress is lost; the SDK warns when
  you close with optimizer steps newer than your last save.
- Creating a run provisions a trainer, usually in under a minute. `runs.create(wait=True)` (the default) blocks
  and logs progress on the `endor` logger; with `wait=False` the run returns at once and `run.ready` is the future.
- If the trainer's GPU dies, pending calls fail with `trainer_lost` and the run becomes `failed` (`run.info().failure`
  says why). Start a new run with `from_model=` your last saved model.

## Teacher distillation

Label unlabeled rows with a teacher of your choice, then fine-tune on its answer distributions:

```python
from endor.recipes import DistillConfig, SupervisedConfig, distill

class MyTeacher:
    def label(self, rows):                        # one {question name: label} per row
        return [{"department": {"billing": 0.7, "technical": 0.3, "sales": 0.0}} for _ in rows]

cfg = DistillConfig(SupervisedConfig(project="tickets", model_name="distilled"), teacher=MyTeacher())
result = distill.distill(cfg, unlabeled_rows, eval_rows=gold_rows)
```

## Download a trained model

A saved model is a LoRA adapter (PEFT) for its base model. Download its files to run it yourself:

```python
files = project.models.download("v1", "./tickets-v1")   # include_optimizer=True adds optimizer.pt
```

This writes `adapter_model.safetensors`, `adapter_config.json`, `endor_manifest.json` and, if the model has one,
`readout.safetensors` into the folder. `optimizer.pt` (Adam state) is only needed to resume training. Each file comes
straight from storage over a link that expires after 10 minutes, and is checked against its size and SHA-256 (when
the API gives one); a file that fails raises `DownloadError` and is not left behind. The links are never returned or
logged, and your API key is never sent to storage. From the shell: `endor models download tickets/v1 -o ./tickets-v1`.

## Errors and retries

Every API error is an `endor.APIError` subclass named after its status (`NotFoundError`, `ConflictError`,
`UnprocessableEntityError`, `RateLimitError`, ...) with the server's stable `code` (`unknown_model`,
`invalid_datum`, `seq_conflict`, ...), the offending `param` when there is one, and the `request_id`.
`InsufficientBalanceError` (402, `insufficient_balance`) means your org's balance is used up: add credit in the
dashboard. A few codes have their own subclass: `LimitReachedError` (409 `limit_reached`) and `NoBaseModelError`
(409 `no_base_model`), both `ConflictError`s, and `ModelRequiresProjectError` (422 `model_requires_project`), an
`UnprocessableEntityError`. A training operation that fails on the trainer raises `OperationFailedError` from `.result()`, with codes
such as `oom`, `no_gradients` or `trainer_lost`. Connection problems raise `APIConnectionError` and
`APITimeoutError`.

Requests that fail with a connection error, a timeout, a 408, a 429 or a 5xx are retried with exponential backoff
and `Retry-After`, except `limit_reached` and `insufficient_balance`, which waiting doesn't fix. Configure it with
`RetryPolicy`:

```python
client = endor.EndorClient(retry=endor.RetryPolicy(max_retries=5, timeout=120))
client = endor.EndorClient(retry=endor.RetryPolicy(max_retries=0))      # no retries
```

Retries are safe: creates carry an `Idempotency-Key`, and a run's compute calls carry a sequence number the server
checks, so a retried request never runs twice.

## Configuration

| Argument | Environment variable | Default |
|---|---|---|
| `api_key` | `ENDOR_API_KEY` | required |
| `base_url` | `ENDOR_BASE_URL` | production |
| `model` | `ENDOR_DEFAULT_MODEL` | none: pass `model=` to `system_one`, e.g. `"tickets"` |
| `timeout` | | 30 s per request (a future poll also waits up to 25 s on the server) |
| | `ENDOR_LOG_LEVEL` | unset; `info` logs one line per request on the `endor` logger |

`EndorClient` also takes `headers` for every request, `retry`, `gzip=False` to stop compressing large bodies, `capture=False`
to stop sending each run's settings and your git commit hash to the dashboard, and your own `httpx` client or
transport.

## What the SDK sends

Besides your API key and the request itself, every request carries:

- `User-Agent: endor-python/<version> (python <version>; <os>; <cpu>)`, with `; cli` added for the `endor` command;
- `X-Endor-SDK: endor-python`, `X-Endor-SDK-Version: <version>`, `X-Endor-Runtime: python/<version> (<os>; <cpu>)`
  and `X-Endor-SDK-Interface: python` or `cli`;
- `X-Endor-SDK-Method` (the SDK call, e.g. `run.forward_backward`), `X-Endor-SDK-Recipe` (inside a recipe),
  `X-Endor-Client-Request-Id` (the same on every retry of one call) and `X-Endor-Retry-Count` (on retries).

`runs.create` also sends the run's settings and, unless `capture=False`, your git commit hash and a dirty flag
(never file contents).

For every request, the API stores the request id, your org, the calling key, the route, the status and the latency,
plus a `client` record with one field per header: `user_agent`, `sdk`, `sdk_version`, `runtime`, `interface`,
`method`, `recipe`, `client_request_id` and `retry_count` (each cut to 200 characters). It also keeps the request
body, and the response body when it is JSON and at most 256 KB, in object storage (S3). The secret API key is never
stored. You can add your own headers with `EndorClient(headers=...)`; the ones above can't be overridden.

## CLI

```
endor whoami
endor base-models
endor projects list | create NAME [--base-model BASE] | delete NAME
endor datasets list PROJECT | upload PROJECT NAME FILE | delete PROJECT NAME
endor runs list PROJECT | show RUN_ID | close RUN_ID
endor models list PROJECT | info MODEL | download MODEL [-o DIR] [--include-optimizer] | set-ttl MODEL SECONDS|none | delete MODEL
endor eval PROJECT MODEL DATASET
endor usage --start 2026-10-01 --end 2026-10-06 [--project P] [--csv]
```

`MODEL` is `<project>/<name>` (`<project>/base` for the base model). Every command takes `-f json`. Training itself happens in Python. `endor runs close`
frees a slot when a script left a run open. `usage --csv` columns are `hour, kind, project, base_model, model,
training_run_id, input_tokens, gpu_seconds, continuous_learning, price_per_mtok, base_cost_usd,
continuous_learning_cost_usd, cost_usd`.

## Reference

| Object | Members |
|---|---|
| `EndorClient(*, api_key, model, retry, timeout, headers, base_url, capture, gzip, http_client, async_http_client, transport, async_transport)` | `system_one(state, questions, *, model, retry, timeout, extra_headers, extra_body, response_model)`, `system_one_async`, `models.list()`, `models.list_async()`, `base_models()`, `projects`, `whoami()`, `usage(starting_on, ending_before, project=None)`, `close()`, `aclose()`, context managers |
| `client.projects` | `create(name, description=None, continuous_learning=None, *, base_model=None)`, `get(name)`, `get_or_create(name, description=None, continuous_learning=None, *, base_model=None)`, `list(limit=None, offset=0)` (all pages) |
| `Project` | `.name`, `.info_`, `.datasets`, `.runs`, `.models`, `info()`, `update(*, description=None, base_model=None, auto_promote=None, base_keep_warm=None, continuous_learning=None)`, `set_live(model)`, `evaluate(model, dataset, run_id=None)` → `APIFuture[Evaluation]`, `evaluations(run_id=None, model=None)`, `delete()` |
| `project.datasets` | `upload(name, rows)`, `list()`, `get(name)`, `rows(name, page=500)`, `delete(name)` |
| `project.runs` | `create(base_model=None, *, rank, alpha, seed, train_attn, train_mlp, train_readout, from_model, include_optimizer, name, tags, config, user_metadata, wait)` → `Run`, `get(run_id)`, `list(tag=None, limit=None, offset=0)` |
| `project.models` | `list(run_id=None)`, `get(model)`, `set_ttl(model, ttl_seconds)`, `set_keep_warm(model, on=True)`, `download(model, path, include_optimizer=False)`, `delete(model)`; `model` is a name or `<project>/<name>`, `base` for the base model |
| `Run` | `.id`, `.project`, `.info_`, `.ready`, `.next_seq_id`, `forward(data, loss_fn)`, `forward_backward(data, loss_fn)`, `optim_step(adam_params=None, *, learning_rate=None)`, `save_checkpoint(name, *, include_optimizer, ttl_seconds, user_metadata)` → `APIFuture[str]`, the `_async` variants of those four, `close(*, wait=False, timeout=None)`, `close_async(...)`, `info()`, `log(metrics, step=None)`, `metrics(keys=None, since_step=None)`, `log_eval(model, results, *, step, name)`, sync and async context manager |
| `APIFuture[T]` | `result(timeout=None)`, `await f`, `result_async(timeout)`, `done()`, `info`, `cancel()`, `cancel_async()`, `APIFuture.completed(value)`; `endor.gather(*futures)`, `endor.gather_async(*futures)` |
| `endor.data` | `to_row`, `load_rows(path)`, `save_rows(path, rows)`, `label_target(question, label)`, `row_to_datums(row)`, `rows_to_datums(rows)`, `split(rows, holdout=0.1, seed=0)`, `batches(items, size, *, shuffle=True, seed=0)` |
| `endor.metrics` | `decision_metrics(probs, targets, bins=10)` → `{n, accuracy, nll, brier, ece, mean_confidence, selective}` |
| `endor.recipes.supervised` | `SupervisedConfig`, `SupervisedResult`, `train(cfg, rows, eval_rows=None, client=None)`, `evaluate_run(run, datums)`, `evaluate_model(client, model, rows)`, `lr_at(cfg, step, total)` |
| `endor.recipes.distill` | `Teacher` (protocol), `DistillConfig(supervised, teacher, budget, seed)`, `label_with_teacher(rows, cfg)`, `distill(cfg, unlabeled_rows, eval_rows, client=None)` |
| Types | `Noul`, `Choice`, `Score`, `NoulCriteria`; `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `SystemOneResponse`, `Usage`, `ModelMetadata`, `ListModelsResponse`, `BaseModelInfo`; `Datum`, `DecisionRow`, `Target`, `LoraConfig`, `AdamParams`, `ForwardOutput`, `OptimStepOutput`; `ProjectInfo`, `ContinuousLearning`, `RunInfo`, `LoraInfo`, `ModelInfo`, `DatasetInfo`, `Evaluation`, `MetricPoint`, `DownloadedFile`, `UsageRow`, `WhoAmI`; helpers `option_keys`, `answer_probabilities`, `question_dict` |
| Errors | `EndorError`; `APIError` with `BadRequestError`, `AuthenticationError`, `InsufficientBalanceError`, `PermissionDeniedError`, `NotFoundError`, `ConflictError` (`LimitReachedError`, `NoBaseModelError`), `PayloadTooLargeError`, `UnprocessableEntityError` (`ModelRequiresProjectError`), `RateLimitError`, `OverloadedError`, `InternalServerError`, `ResponseValidationError`; `APIConnectionError`, `APITimeoutError`; `OperationFailedError` |

Every public member has a docstring with the details.

## License

MIT
