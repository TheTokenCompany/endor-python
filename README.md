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

## Decisions

```python
import endor
from endor import Choice, Noul, Score

client = endor.EndorClient()

res = client.system_one(
    {"subject": "Charged twice", "body": "I was billed two times for my March invoice."},
    {
        "department": Choice(instructions="Which team should handle `body`?",
                             criteria={"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": None}),
        "urgent": Noul(instructions="Does the customer need an answer today?"),
        "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"]),
    },
    model="tickets/v1",          # a base model id, or one of your models as "<project>/<name>"
)

res.choices["department"].choice          # "billing"
res.choices["department"].probabilities   # {"billing": 0.91, "technical": 0.07, "sales": 0.02}
res.nouls["urgent"].noul                  # 0.83
res.scores["frustration"].score           # 1.4 (expected level), plus .probabilities and .legend
res.model, res.usage.input_tokens
```

Questions can also be plain dicts in the same shape. Every question is answered independently against the same
state. Refer to parts of the state with backticked paths, as in ``"Is `body` angry?"``.

**Typed responses.** Subclass `SystemOneResponse` to get one attribute per question:

```python
from endor import ChoiceAnswer, NoulAnswer, SystemOneResponse

class Triage(SystemOneResponse):
    department: ChoiceAnswer
    urgent: NoulAnswer

triage = client.system_one(state, questions, response_model=Triage)
triage.department.choice, triage.urgent.noul
```

**The catalog.** `client.models.list()` returns the base models and your saved models; `client.base_models()`
returns the base models with their option limits and prices.

**Async.** `await client.system_one_async(...)` and `await client.models.list_async()`; `async with EndorClient()`
closes the connections.

## Projects

Everything you create lives in a **project**, keyed by its name. A project holds datasets, runs, the models those
runs save (named `<project>/<name>`), and evaluations.

```python
project = client.projects.get_or_create("tickets")
project.datasets.upload("train", endor.data.load_rows("train.jsonl"))   # see docs/DATA_FORMAT.md
project.datasets.upload("heldout", endor.data.load_rows("heldout.jsonl"))
```

A dataset row is a decision request with labels:

```json
{"state": {"body": "I was charged twice"},
 "questions": {"department": {"type": "choice", "criteria": {"billing": null, "technical": null}},
               "urgent": {"type": "noul", "instructions": "Does this need an answer today?"}},
 "labels": {"department": "billing", "urgent": true}}
```

Labels can be hard (`"billing"`, `true`, a level index) or soft (a distribution over the options). The full format,
with the option ids every label and probability is keyed by, is in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Fine-tune in a few lines

```python
from endor.recipes import SupervisedConfig, supervised

result = supervised.train(SupervisedConfig(project="tickets", model_name="v1"), endor.data.load_rows("train.jsonl"))

result.base_metrics["accuracy"], result.final_metrics["accuracy"]   # held-out: the base model, then yours
client.system_one(state, questions, model=result.model)             # "tickets/v1"
```

The recipe holds out 10% of the rows, scores the untuned base model on them, trains one epoch (learning rate `1e-4`
with warmup then linear decay, batches of 16), scores the held-out rows during and after training, and saves the
final model. Every number appears on the run's dashboard page. `SupervisedConfig` has the knobs: base model, rank,
learning rate and schedule, batch size, epochs, loss, how often to evaluate.

## Or drive the loop yourself

```python
rows = endor.data.load_rows("train.jsonl")
train, heldout = endor.data.split(rows, holdout=0.1)
train, heldout = endor.data.rows_to_datums(train), endor.data.rows_to_datums(heldout)

with project.runs.create(base_model="pplx-decider-v1-27b", rank=16) as run:
    for batch in endor.data.batches(train, 16):
        fb = run.forward_backward(batch)            # gradients accumulate on the trainer
        opt = run.optim_step(learning_rate=1e-4)    # AdamW step, then zero gradients
        out, step = endor.gather(fb, opt)           # submit both, then wait once
    scored = run.forward(heldout).result()          # option probabilities with the current adapter
    metrics = endor.metrics.decision_metrics(scored.probabilities, [d.target for d in heldout])
    run.log({"heldout/accuracy": metrics["accuracy"]})
    model = run.save_checkpoint("v2", include_optimizer=True).result()   # "tickets/v2", usable at once

resumed = project.runs.create(from_model="v2", include_optimizer=True)   # continue exactly where v2 stopped
```

- A `Datum` is one state, one question, a `Target` (a hard `label` or soft `probs`) and a weight. Each labeled
  question of a row becomes one datum.
- `forward`, `forward_backward`, `optim_step` and `save_checkpoint` return futures. Call `.result()`, `await` them,
  or wait for several with `endor.gather(...)`. Each has an `_async` variant that submits without blocking.
- Gradients are the weighted mean over every datum since the last `optim_step`, so splitting a batch across calls
  changes nothing. Batches over 1,024 datums are split for you.
- Losses: `cross_entropy` (hard or soft targets) and `brier`.
- `optim_step` takes `AdamParams` (betas, epsilon, weight decay, gradient clipping) and the learning rate, which you
  pass every step: the schedule is yours.
- `run.forward` scores data with the current adapter without saving a model. `project.evaluate(model, dataset)`
  scores a saved model on a stored dataset server-side.
- `run.close()` releases the trainer (a `with` block does it for you). Unsaved progress is lost; the SDK warns when
  you close with optimizer steps newer than your last save.
- Creating a run provisions a trainer, which can take minutes. `runs.create(wait=True)` (the default) blocks and
  logs progress on the `endor` logger; with `wait=False` the run returns at once and `run.ready` is the future.

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

## Errors and retries

Every API error is an `endor.APIError` subclass named after its status (`NotFoundError`, `ConflictError`,
`UnprocessableEntityError`, `RateLimitError`, ...) with the server's stable `code` (`unknown_model`,
`invalid_datum`, `seq_conflict`, ...), the offending `param` when there is one, and the `request_id`. A training
operation that fails on the trainer raises `OperationFailedError` from `.result()`, with codes such as `oom` or
`trainer_lost`. Connection problems raise `APIConnectionError` and `APITimeoutError`.

Requests that fail with a connection error, a timeout, a 408, a 429 or a 5xx are retried with exponential backoff
and `Retry-After`. Configure it with `RetryPolicy`:

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
| `model` | `ENDOR_DEFAULT_MODEL` | the current base model |
| `timeout` | | 30 s per request |
| | `ENDOR_LOG_LEVEL` | unset; `info` logs one line per request on the `endor` logger |

`EndorClient` also takes `headers` for every request, `retry`, `gzip=False` to stop compressing large bodies, `capture=False`
to stop sending each run's settings and your git commit hash to the dashboard, and your own `httpx` client or
transport. Every request identifies the SDK, its version and the SDK method behind it; the headers are listed in
[docs/REQUEST_HEADERS.md](docs/REQUEST_HEADERS.md).

## CLI

```
endor whoami
endor base-models
endor projects list | create NAME | delete NAME
endor datasets list PROJECT | upload PROJECT NAME FILE | delete PROJECT NAME
endor runs list PROJECT | show RUN_ID | close RUN_ID
endor models list PROJECT | info MODEL | download MODEL [-o FILE] | set-ttl MODEL SECONDS|none | delete MODEL
endor eval PROJECT MODEL DATASET
endor usage --start 2026-10-01 --end 2026-10-06 [--project P] [--csv]
```

`MODEL` is `<project>/<name>`. Every command takes `-f json`. Training itself happens in Python.

## Reference

| Object | Members |
|---|---|
| `EndorClient(*, api_key, model, retry, timeout, headers, base_url, capture, gzip, http_client, async_http_client, transport, async_transport)` | `system_one(state, questions, *, model, retry, timeout, extra_headers, extra_body, response_model)`, `system_one_async`, `models.list()`, `models.list_async()`, `base_models()`, `projects`, `whoami()`, `usage(start, end, project=None)`, `close()`, `aclose()`, context managers |
| `client.projects` | `create(name, description=None)`, `get(name)`, `get_or_create(name, description=None)`, `list(limit, offset)` |
| `Project` | `.name`, `.info_`, `.datasets`, `.runs`, `.models`, `info()`, `evaluate(model, dataset, run_id=None)` → `APIFuture[Evaluation]`, `evaluations(run_id=None, model=None)`, `delete()` |
| `project.datasets` | `upload(name, rows)`, `list()`, `get(name)`, `rows(name, page=500)`, `delete(name)` |
| `project.runs` | `create(base_model=None, *, rank, alpha, seed, train_attn, train_mlp, train_readout, from_model, include_optimizer, name, tags, config, user_metadata, wait)` → `Run`, `get(run_id)`, `list(tag, limit, offset)` |
| `project.models` | `list(run_id=None)`, `get(model)`, `set_ttl(model, ttl_seconds)`, `archive_url(model)`, `delete(model)`; `model` is a name or `<project>/<name>` |
| `Run` | `.id`, `.project`, `.info_`, `.ready`, `.next_seq_id`, `forward(data, loss_fn)`, `forward_backward(data, loss_fn)`, `optim_step(adam_params=None, *, learning_rate=None)`, `save_checkpoint(name, *, include_optimizer, ttl_seconds, user_metadata)` → `APIFuture[str]`, the `_async` variants of those four, `close()`, `close_async()`, `info()`, `log(metrics, step=None)`, `metrics(keys=None, since_step=None)`, `log_eval(model, results, *, step, name)`, context manager |
| `APIFuture[T]` | `result(timeout=None)`, `await f`, `result_async(timeout)`, `done()`, `info`, `cancel()`, `cancel_async()`, `APIFuture.completed(value)`; `endor.gather(*futures)`, `endor.gather_async(*futures)` |
| `endor.data` | `to_row`, `load_rows(path)`, `save_rows(path, rows)`, `label_target(question, label)`, `row_to_datums(row)`, `rows_to_datums(rows)`, `split(rows, holdout=0.1, seed=0)`, `batches(items, size, *, shuffle=True, seed=0)` |
| `endor.metrics` | `decision_metrics(probs, targets, bins=10)` → `{n, accuracy, nll, brier, ece, mean_confidence, selective}` |
| `endor.recipes.supervised` | `SupervisedConfig`, `SupervisedResult`, `train(cfg, rows, eval_rows=None, client=None)`, `evaluate_run(run, datums)`, `evaluate_model(client, model, rows)`, `lr_at(cfg, step, total)` |
| `endor.recipes.distill` | `Teacher` (protocol), `DistillConfig(supervised, teacher, budget, seed)`, `label_with_teacher(rows, cfg)`, `distill(cfg, unlabeled_rows, eval_rows, client=None)` |
| Types | `Noul`, `Choice`, `Score`, `NoulCriteria`; `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `SystemOneResponse`, `Usage`, `ModelMetadata`, `ListModelsResponse`, `BaseModelInfo`; `Datum`, `DecisionRow`, `Target`, `LoraConfig`, `AdamParams`, `ForwardOutput`, `OptimStepOutput`; `ProjectInfo`, `RunInfo`, `ModelInfo`, `DatasetInfo`, `Evaluation`, `MetricPoint`, `ArchiveInfo`, `UsageRow`, `WhoAmI`; helpers `option_keys`, `answer_probabilities`, `question_dict` |
| Errors | `EndorError`; `APIError` with `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `ConflictError`, `PayloadTooLargeError`, `UnprocessableEntityError`, `RateLimitError`, `OverloadedError`, `InternalServerError`, `ResponseValidationError`; `APIConnectionError`, `APITimeoutError`; `OperationFailedError` |

Every public member has a docstring with the details.

## License

MIT
