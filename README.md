<div align="center">

# Endor Python SDK

Decision models that learn from your decisions, and custom models you train yourself.

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

The usual way to use Endor is a **continuous learning project**: you pick a base model, send it decisions, tell it
when it was wrong, and it gets better at your task by itself. For full control, you can also train **custom models**
with the SDK (see "Advanced: custom models").

## Install

```bash
pip install endor
export ENDOR_API_KEY=edk_...
```

Python 3.10 and up. Dependencies: `httpx` and `pydantic`.

The SDK calls production (`https://api.endor.thetokencompany.com`). For another environment, such as staging, set
`ENDOR_BASE_URL=https://staging.api.endor.thetokencompany.com` or pass `EndorClient(base_url=...)`.

API keys belong to an organization (`edk_...`, created in the dashboard); everything you create belongs to that org.

## Continuous learning projects

Create a project on a base model, then send it decisions. It answers at once with the base model and learns from its
own decisions and your corrections, making new versions that you promote to production (or that go there by
themselves with auto-promote on):

```python
import endor
from endor import Choice, Noul, Score

client = endor.EndorClient()
project = client.projects.get_or_create("tickets", base_model="mapika/decider-2b")   # the base model is chosen once

res = client.system_one(
    {"subject": "Charged twice", "body": "I was billed two times for my March invoice."},
    {
        "department": Choice(instructions="Which team should handle `body`?",
                             criteria={"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": None}),
        "urgent": Noul(instructions="Does the customer need an answer today?"),
        "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"]),
    },
    model="tickets",             # the project's production version
)

res.choices["department"].choice          # "billing"
res.choices["department"].probabilities   # {"billing": 0.91, "technical": 0.07, "sales": 0.02}
res.nouls["urgent"].noul                  # 0.83
res.scores["frustration"].score           # 1.4 (expected level), plus .probabilities and .legend
res.model, res.usage.input_tokens         # what answered: "tickets:v4", or "tickets" while its base model answers
res.id                                    # the decision's id ("req_..."), to correct it later
```

**Production and preview, like deployments.** A project has one **production** version. A new version lands in the
**preview** and waits there until you promote it to production. Turn **auto-promote** on (it is off by default) and
every new version goes to production as soon as it is ready instead:

| `model` | Answers with |
|---|---|
| `"tickets"` (`project.ref`) | the production version (`project.production`); the base model until you promote one |
| `"tickets:preview"` (`project.preview_ref`) | the preview (`project.preview`), the newest version while it isn't in production. Without one (always while auto-promote is on) it raises `NoPreviewError`, and `project.preview_ref` is None |
| `"router-2"` | a custom model in production (see "Advanced: custom models") |

Promote any version yourself, for example to roll back:

```python
project.versions()               # v1, v2, ... newest first: .production, .preview (never both), .accuracy (%), .loss
project.promote("v3")            # v3 to production; a version older than the newest turns auto-promote off
project.set_auto_promote(True)   # the newest goes to production at once, and every new version after it (off by default)
project.info()                   # .production, .preview, .auto_promote, .promoted_at, .versions, .base_model
project.production, project.preview, project.auto_promote   # the same, as last fetched
project.download("v3", "./tickets-v3")   # a version's LoRA adapter (PEFT), to run it yourself
```

Older versions can't be called directly (`"tickets:v3"` raises `VersionNotCallableError`): promote one to serve it.

**Its learning.** `project.learning()` shows how it is going:

```python
learning = project.learning()
learning.status, learning.progress    # "learning", 0.4 of the way to the next version
learning.serving                      # "tickets:v2", or "tickets" while its base model answers
for t in learning.trainings:          # newest first: kept or not, with its accuracy and loss
    print(t.version, t.kept, t.decisions, t.accuracy, t.against_accuracy)
learning.drift                        # per question, whether recent traffic changed
```

A version's and a training's `accuracy` is in percent (`87.5`); custom models and evaluations give accuracy from 0 to
1 (`0.875`).

A project's decisions are billed at the base model's continuous-learning price (`client.base_models()`:
`price_per_mtok_decide_continuous_learning`). Continuous learning is turned on per organization:
`client.whoami().continuous_learning` says whether yours has it (while it doesn't, projects and their decisions raise
`PermissionDeniedError`, 403 `continuous_learning_not_enabled`). An organization has up to 5 projects
(`client.whoami().limits`). Names
are lowercase letters, digits, `.`, `_` or `-`, up to 63 characters, unique in the organization together with your
custom models. `client.projects.list()`, `client.projects.get(name)`, `project.update(description=...)` and
`project.delete()` do what they say.

### Corrections

When you learn whether a decision was right, correct it. For each question you checked, say whether the model
answered right and, if not, the right answer. The project learns from your corrections in its next versions:

```python
client.corrections.create(res, {
    "frustration": {"correct": True},                       # the model was right
    "department": {"correct": False, "answer": "technical"},  # wrong: the right answer
    "urgent": {"correct": False},                           # wrong: a yes/no question, so the other answer
}, comment="a bug report, not a billing question")
```

- An `answer` must be exactly one of the question's options as the decision asked it: a choice's option id (case and
  spelling included), a noul's `True` or `False`, a score's level index as an `int`. Anything else, an "other"
  answer included, raises `UnprocessableEntityError` (422, `param` `questions.<name>.answer`). When none of the
  options fits, send `{"correct": False}`.
- `{"correct": False}` alone names the right answer for a noul, or a choice or score of two options. With more
  options the correction's `answer` is `None`: only known to be wrong.
- A decision has one correction: creating it again replaces it whole. `client.corrections.get(res)` reads it,
  `client.corrections.delete(res)` withdraws it, and `client.corrections.batch([...])` sends up to 1,000 at once,
  all or nothing (`{"request_id", "questions", "comment"?}` each), for example to backfill decisions you checked
  earlier. Each has an `_async` variant. A decision is its `SystemOneResponse` or its id (`res.id`).
- Corrections are free; only continuous learning projects learn from them.
- Endor records a decision just after it answers, so a correction sent at once can find no decision yet. `create` and
  `batch` then try again for up to 5 seconds before they raise `NotFoundError` (404 `not_found`), as they do for a
  decision id Endor doesn't know.

## Decisions

`model` is `"<project>"`, `"<project>:preview"` (while the project has a preview), the name of a custom model in
production, or a base model's id. There is no default model: pass `model=` or set one with
`EndorClient(model="tickets")` or `ENDOR_DEFAULT_MODEL`, else `system_one` raises `EndorError` before sending anything.

**Base models directly.** Call a base model by its id, `owner/model` as on Hugging Face (`client.base_models()` lists
them), at its listed price, `price_per_mtok_decide`:

```python
client.system_one(state, questions, model="mapika/decider-2b")
```

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

**Benchmarks.** Pass `exclude_from_training=True` to `system_one` (or `EndorClient(exclude_from_training=True)`
for every call) to keep decisions out of a project's learning, for example a benchmark or an evaluation. They are
still answered, billed and logged.

**Images.** On a base model that reads images (`client.base_models()`: `modalities` includes `"image"`), put
`endor.Image` objects anywhere in the state. PNG, JPEG or WebP, at most 8 per decision and 5 MB each; their tokens
are billed like text. Other base models answer `422 unsupported_modality`.

```python
state = {"screenshot": endor.Image.from_path("checkout.png"), "goal": "Pay for the order"}
client.system_one(state, {"pay": Noul(instructions="Is the Pay button visible?")}, model="shop")
```

`Image.from_bytes(data, media_type=None)` and `Image.from_pil(image, format="PNG")` (needs Pillow) build one too.
Images are for decisions only: training data (datasets, datums, evaluation and prediction rows) is text and JSON, and
the API refuses images there (422).

**The catalog.** `client.models.list()` returns your own names you can pass as `model` (`GET /v1/models`): for each
project `"<project>"` and, while it has a preview, `"<project>:preview"`, then every custom model in production (`.kind` is
`project` or `custom`; `.environment` is `production` or `preview` for a project's names). `client.base_models()` returns the base models (`GET /v1/base_models`) with their option limits,
`modalities`, `hf_repo`, `contract` and prices (`price_per_mtok_decide` for the base itself and custom models,
`price_per_mtok_decide_continuous_learning` for projects, `price_per_gpu_hour` for training).

To choose a base model, compare what each is meant for, its ranks, speed, context and price:

```python
for b in client.base_models():
    i = b.decision_index                  # its entry on the Jev Decision Index, None if it isn't there
    print(b.id, i and i.meant_for,        # ["Code", "Agents", ...], best first; empty when none
          i and f"{i.rank}/{i.entrants}", # its rank on the board
          b.tokens_per_s,                 # decision input tokens per second Endor measured, None if not measured
          b.max_question_tokens, b.price_per_mtok_decide)
```

`decision_index` also has `tied` (within the board's tie band of another entry), `use_case_ranks` (its mean rank on
each use case's benchmarks), `task_ranks` (its rank on each public benchmark, by what it tests, for example
`"phishing detection"`), `note` (where the index ran another form of the weights than Endor serves) and `url` (its
scores on the index). `meant_for` holds `"Code"`, `"Agents"`, `"Documents and email"` and `"Finance"` where its mean
rank on that use case is in the board's top quarter. `endor base-models` prints the same as a table.

**Many rows at once.** `client.predict(model, rows)` returns the answers to every question of every row, in the rows'
order, from one server-side evaluation of them: nothing to upload, nothing scored, billed as the same decisions.
`model` is a custom model's name (in production or archived) or a base model's id. Rows are `{"state", "questions"}` (any shape
`client.datasets.upload` takes; labels are ignored), at most 50,000 and 32 MB per call. While it runs it prints a
progress line every 10 seconds (`progress=False` turns it off); `timeout=` bounds the wait. An unknown model raises
`NotFoundError` (404 `unknown_model`).

```python
answers = client.predict("router-2", [{"state": t, "questions": questions} for t in new_tickets])
answers[0]["department"].choice           # "billing": each row's answers as system_one(...).answers has them
```

**Async.** `await client.system_one_async(...)`, `await client.predict_async(...)` and
`await client.models.list_async()`; `async with EndorClient()` closes the connections. One client works across
several `asyncio.run()` calls: each event loop gets its own connections.

### Data format

Datasets, evaluations and predictions take decision rows. A row has `state` (text or JSON), `questions` (named
`Noul`, `Choice` or `Score` objects, as `system_one` takes), and optionally `labels` (question name → label), `id` and
`weight` (default 1). Up to 50,000 rows per dataset. Each labeled question becomes one training example; unlabeled
questions are skipped.

```json
{"state": {"body": "I was charged twice"},
 "questions": {"department": {"type": "choice", "criteria": {"billing": null, "technical": null}},
               "urgent": {"type": "noul", "instructions": "Does this need an answer today?"}},
 "labels": {"department": "billing", "urgent": true}}
```

Every option has an id, used in labels, training targets and answer probabilities: `"false"`/`"true"` for a noul,
the `criteria` keys for a choice, `"0"`…`"n-1"` (the level index) for a score.

| Type | Hard label | Soft label (every option, non-negative, summing to 1 ± 0.02) |
|---|---|---|
| `noul` | `true` / `false` | `0.8` (P(true)) or `{"true": 0.8, "false": 0.2}` |
| `choice` | `"billing"` | `{"billing": 0.7, "technical": 0.2, "sales": 0.1}` |
| `score` | `2` (level index) or `"2"` (its option id) | `[0.1, 0.6, 0.3]` or `{"0": 0.1, "1": 0.6, "2": 0.3}` |

A soft label is the probability map itself. `{"probs": {...}}` is the shape of a `Datum` target (below), not of a
row label, and is refused with a ValueError that says so.

`endor.data.load_rows(path)` reads `.jsonl` or a `.json` array and also accepts single-question rows,
`{"state", "question", "expected"}` or `{"state", "question", "label"?, "target"?}`, each becoming one question named
`decision`. `endor.data.save_rows(path, rows)` writes `.jsonl`.

## Advanced: custom models

A custom model is one you train yourself with the SDK, on a base model, from your own labeled data. Each save is a new
model with its own name, unique in the organization together with your projects. A saved model starts in the
**archive**: it can be downloaded, evaluated and trained from, and decisions can call it by its name once you
**promote** it to **production**:

```python
with client.custom_models.create("router-2", base_model="mapika/decider-2b") as run:   # a run that saves as "router-2"
    ...                                          # forward_backward, optim_step (below), or use the recipe
    run.save_checkpoint().result()               # "router-2", in the archive
client.custom_models.promote("router-2")         # production: model="router-2" answers
client.system_one(state, questions, model="router-2")
client.custom_models.archive("router-2")         # back to the archive: decisions refuse it
```

- `client.custom_models.list()` lists your custom models, the ones in production first (`production=True` or
  `production=False` keeps one of the two); `get(name)` has `state` (`"production"` or `"archived"`), `production`,
  `promoted_at`, `base_model`, `step`, and `loss` and `accuracy` (0 to 1): the run's training loss and accuracy at the
  model's step.
- A model in production has `serving`: `"deploying"` right after `promote` (decisions are accepted; the first ones
  may wait while a decision server loads it), then `"live"`, or `"failed"` with `serving_error` saying why
  (`promote(name)` again retries). It is None while archived. `endor custom-models list` and `info` show it.
- An organization has a few models in production at once (`max_production_models_per_org` in
  `client.whoami().limits`). When every slot is taken, `promote` raises `ProductionFullError` naming the models in
  production; `promote(name, replace="router-1")` archives that one in the same step. `archive(name)` frees a slot.
  Ask us for more production slots.
- Decisions refuse an archived model with `ModelNotInProductionError` (409 `model_not_in_production`). Models in
  production never expire: `set_ttl(name, seconds)` gives an archived model a lifetime, and `delete(name)` removes an
  archived one (`ModelInProductionError` for one in production: archive it first).
- Datasets (`client.datasets`), runs (`client.runs`) and evaluations (`client.evaluations`) belong to the
  organization. `client.runs.create(base_model, model="router-2", ...)` is the same as
  `client.custom_models.create("router-2", base_model, ...)`.

```python
client.datasets.upload("train", endor.data.load_rows("train.jsonl"))     # rows: see "Data format"
client.datasets.upload("heldout", endor.data.load_rows("heldout.jsonl"))
client.evaluations.create("mapika/decider-2b", "heldout").result()       # the untuned base, as a baseline
client.evaluations.create("router-2", "heldout").result()                # a custom model, in production or not
client.evaluations.create("router-2", rows=more_rows).result()          # or rows sent with the call, no dataset
```

An evaluation scores a row's labeled questions; a row without labels is answered on every question and not scored
(its answers are in the results' `rows_url`, which `client.predict` reads for you). Its accuracies are from 0 to 1.
`client.evaluations.get(evaluation_id)` fetches one again, with `rows_done` of `rows_total` while it runs, and
`client.evaluations.list(run_id=None, model=None)` lists them. An unknown model raises `NotFoundError` (404
`unknown_model`).

**Model names** are lowercase letters, digits, `.`, `_` or `-`, starting with a letter or digit, up to 63 characters,
and can't be a base model id. `save_checkpoint` raises `ValueError` for a bad name before sending it. Each name is
saved once: a name a custom model or project already has raises `ConflictError` (409 `conflict`) at once. The run
itself is unaffected, but an error you don't catch leaves the `with` block, which closes the run. For checkpoints
during a long run, give each its own name, e.g. `run.save_checkpoint(f"router-ckpt-{step}", ttl_seconds=86_400)`.

### Train in a few lines

```python
from endor.recipes import SupervisedConfig, supervised

cfg = SupervisedConfig(model_name="router-2")   # base_model="mapika/decider-2b" by default
result = supervised.train(cfg, endor.data.load_rows("train.jsonl"))

result.base_metrics["accuracy"], result.final_metrics["accuracy"]   # held-out: the untuned base, then yours
client.custom_models.promote(result.model)                          # to production
client.system_one(state, questions, model=result.model)             # "router-2"
```

The recipe holds out 10% of the rows, scores the untuned base model on them, trains one epoch (learning rate `1e-4`
with warmup then linear decay, batches of 16), scores the held-out rows during and after training, and saves the
final adapter as the custom model `model_name`, in the archive. Every number appears on the run's dashboard page.
`SupervisedConfig` has the knobs: base model, rank, learning rate and schedule, batch size, epochs, loss, how often to
evaluate. The recipe sets the run's `total_steps` from the data size and epochs, so the dashboard shows a progress bar
with an ETA, and it prints a progress line (`step 37/120 (31%) · loss 0.4120 · ~3 min left`; `progress=False` turns it
off). It sends each step before the last one's results are read (`steps_in_flight=2`), so the trainer doesn't wait for
a round trip between steps; results, metrics and evaluations are the same as with `steps_in_flight=1`.

Before training, the recipe prints a warning for each question with options that have no training labels, or fewer
than 10 (`warning: question 'dept': no training labels for refund; the model can't learn an option it never sees`).
Add rows for those options, or drop them from the question.

### Training costs and limits

- **Billing.** Training is billed by the second for the GPU time your training calls use, one price per GPU-hour for
  every base model; start-up, model loading and the time between calls are free, and each run session costs at least
  60 GPU-seconds. Decisions are billed per 1M input tokens, per base model.
  `client.base_models()` shows the prices; `client.usage(starting_on, ending_before)` returns hourly `decide` rows
  (input tokens) and `train` rows (GPU-seconds) with their cost.
- **Idle timeout.** After 2 minutes without calls, a run saves its state and frees its GPU (status `idle`). The next
  call resumes it in a few seconds. While a run is open, the SDK sends a heartbeat every minute from a background
  thread, so the run stays open however long you wait between calls. When the process ended without closing the run,
  Endor closes it an hour later; saved models are kept. To continue it, `client.runs.get(run_id)` gives the run again
  (`run.info_.step` is its step), and that handle sends the heartbeat too until the run is closed or the handle is
  dropped. The handles `client.runs.list()` returns are for looking: they send none.
- **Open runs per org.** Every run that isn't closed counts toward the org's open-run limit, idle ones included: 25
  by default, and `max_active_runs` in `client.whoami().limits` is your org's own (`endor whoami` shows it). One more `runs.create` raises
  `LimitReachedError` (409, code `limit_reached`), whose message lists the open runs; close one (`run.close()` or
  `endor runs close RUN_ID`). It is not retried.
- **Other limits.** Projects, custom models (and those in production), datasets and API keys also have limits (`LimitReachedError` once reached).
  `client.whoami().limits` lists every limit with its value.
- **Close your runs.** Use `with client.runs.create(...) as run:` (or `async with`): the run is closed when the block
  ends, also on errors. `runs.create(wait=True)` closes the run if you interrupt it while it provisions, and the
  recipes close their run whatever happens. `run.close()` refuses later calls, lets calls already accepted finish
  (status `closing`), then releases the GPU (status `closed`); it returns at once, or waits with `close(wait=True)`.
- **Balance.** When your org's balance is used up (a new org starts with none), decisions, new runs, training calls
  and server-side evaluations raise `InsufficientBalanceError` (402, code `insufficient_balance`) until you add
  credit in the dashboard under Billing. A call the server already accepted still finishes. `client.balance()` (or
  `endor whoami`) shows what is left, in USD.

### Or drive the loop yourself

```python
rows = endor.data.load_rows("train.jsonl")
train, heldout = endor.data.split(rows, holdout=0.1)
train, heldout = endor.data.rows_to_datums(train), endor.data.rows_to_datums(heldout)

with client.custom_models.create("router-2", base_model="mapika/decider-2b", rank=16) as run:
    steps = endor.Steps()                           # keeps 2 steps in flight
    for batch in endor.data.batches(train, 16):
        fb = run.forward_backward(batch)            # gradients accumulate on the trainer
        opt = run.optim_step(learning_rate=1e-4)    # AdamW step, then zero gradients
        for out, step in steps.add(fb, opt):        # sent; returns the previous step's results
            print(step.step, out.loss)
    for out, step in steps.wait():                  # the last step
        print(step.step, out.loss)
    scored = run.forward(heldout).result()          # option probabilities with the current adapter
    metrics = endor.metrics.decision_metrics(scored.probabilities, [d.target for d in heldout])
    run.log({"heldout/accuracy": metrics["accuracy"]})   # at the run's current step
    model = run.save_checkpoint(include_optimizer=True).result()   # "router-2", in the archive

resumed = client.custom_models.create("router-3", from_model="router-2", include_optimizer=True)  # continue exactly
```

A run created with `from_model` (a custom model's name) inherits that model's base model and LoRA settings (rank,
alpha, which layers are trained); the SDK sends only the settings you pass explicitly, and one that conflicts with the
model is a 422. A fresh run defaults to rank 16, alpha 32, attention and MLP layers, no readout.

- A `Datum` is one state, one question, a `Target` (a hard `label` or soft `probs`) and a weight. Each labeled
  question of a row becomes one datum. A target is `Target(label="billing")` or
  `Target(probs={"billing": 0.7, "tech": 0.3})` (as a dict, `{"probs": {...}}`); the bare map is a row label's shape
  and is refused. `endor.data.label_target(question, label)` turns a row label into a `Target`.
- `forward`, `forward_backward`, `optim_step` and `save_checkpoint` return futures. Call `.result()`, `await` them,
  or wait for several with `endor.gather(...)`. Each has an `_async` variant that submits without blocking. A future
  can stay pending while a GPU starts (usually under a minute); `.result()` keeps waiting.
- `endor.Steps()` sends each step before the last one's results are read, so the trainer starts the next step as soon
  as it finishes one instead of waiting for your round trip (most of a second per step). `steps.add(fb, opt)` returns
  the results of the steps it waited for (the step before, once two are out), oldest first, and `steps.wait()` the
  rest; a failed step raises from the call that reads it. A call you send between steps (`run.forward`,
  `save_checkpoint`) runs after every step added so far. `Steps(in_flight=1)` waits for each step, like
  `endor.gather(fb, opt)` after every step. `add_async` and `wait_async` are for async code.
- Gradients are the weighted mean over every datum since the last `optim_step`, so splitting a batch across calls
  changes nothing. `optim_step` with nothing accumulated fails with `no_gradients`. Batches over 1,024 datums are
  split for you; every datum is checked locally before any chunk is sent, so a bad datum never leaves half a batch
  applied.
- Losses: `cross_entropy` (hard or soft targets) and `brier`.
- `optim_step` takes `AdamParams` (betas, epsilon, weight decay, gradient clipping) and the learning rate, which you
  pass every step: the schedule is yours.
- `run.predict(rows)` answers decision rows with the current adapter before you save it: a future of one
  `{name: answer}` per row, in order, as `client.system_one` answers them (labels are ignored). It runs on the
  training GPU in order with the run's other calls and is billed as training GPU time, like `forward`. Lists over
  1,024 questions are split for you. Text and JSON states only.
- `run.forward` scores data with the current adapter without saving a model. `client.evaluations.create(model,
  dataset)` scores a saved model server-side (or on `rows=` sent with the call).
- `run.close()` releases the trainer (a `with` block does it for you). Unsaved progress is lost; the SDK warns once
  when you close with optimizer steps newer than your last save.
- Creating a run provisions a trainer: under a minute on a warm GPU, a few minutes to start a cold one.
  `runs.create(wait=True)` (the default) blocks and logs progress at INFO on the `endor` logger. Since Python hides
  INFO logs unless you configure logging, it also prints one line to stderr once the wait passes 10 seconds (`run
  run_...: waiting for a GPU (starting a cold one can take a few minutes)…`); `runs.create(progress=False)` (or
  `custom_models.create(..., progress=False)`, `SupervisedConfig(progress=False)`) leaves it out. With `wait=False`
  the run returns at once and `run.ready` is the future.
- If the trainer's GPU dies, pending calls fail with `trainer_lost` and the run becomes `failed` (`run.info().failure`
  says why). Start a new run with `from_model=` your last saved model.
- Progress: pass `runs.create(total_steps=N)` (or later `run.set_total_steps(N)`) and the dashboard shows a progress
  bar with an ETA; without it, the step count and steps per minute. `run.info()` has `step`, `total_steps`,
  `seconds_per_step` (median over the last 20 steps), `progress` (0 to 1) and `eta_seconds`.

### Weights & Biases

Endor logs runs to Weights & Biases from its servers; your training code doesn't change. Connect W&B once in the
dashboard (Settings > Integrations, with a W&B API key that Endor keeps in AWS Secrets Manager), and every run logs to
it; turn it off or on per run with `client.runs.create(..., wandb=False)` or `wandb=True`. Each run becomes a W&B run
with the same id, with the base model, LoRA settings and your `config`, and every run metric (`train/loss`,
`train/accuracy`, `train/grad_norm`, `train/lr`, `run.log`) against the Endor step; it is finished when the run
closes. `run.info().wandb_url` links to it.

### Teacher distillation

Label unlabeled rows with a teacher of your choice, then fine-tune on its answer distributions:

```python
from endor.recipes import DistillConfig, SupervisedConfig, distill

class MyTeacher:
    def label(self, rows):                        # one {question name: label} per row
        return [{"department": {"billing": 0.7, "technical": 0.3, "sales": 0.0}} for _ in rows]

cfg = DistillConfig(SupervisedConfig(model_name="router-distilled"), teacher=MyTeacher())
result = distill.distill(cfg, unlabeled_rows, eval_rows=gold_rows)
```

### Download a trained model

A custom model (in production or archived) or a project's version is a LoRA adapter (PEFT) for its base model. Download its files
to run it yourself:

```python
files = client.custom_models.download("router-2", "./router-2")   # include_optimizer=True adds optimizer.pt
files = project.download("v3", "./tickets-v3")                    # a continuous learning project's version
```

This writes `adapter_model.safetensors`, `adapter_config.json`, `endor_manifest.json` and, if the model has one,
`readout.safetensors` into the folder. `optimizer.pt` (Adam state) is only needed to resume training, and only a model
saved with `run.save_checkpoint(include_optimizer=True)` has it (`CustomModelInfo.has_optimizer`): for any other model,
or a project's version, `include_optimizer=True` raises `DownloadError` before anything is written. Each file comes
straight from storage over a link that expires after 10 minutes, and is checked against its size and SHA-256 (when
the API gives one); a file that fails raises `DownloadError` and is not left behind. The links are never returned or
logged, and your API key is never sent to storage. From the shell: `endor custom-models download router-2 -o ./router-2`.

## Errors and retries

Every API error is an `endor.APIError` subclass named after its status (`NotFoundError`, `ConflictError`,
`UnprocessableEntityError`, `RateLimitError`, ...) with the server's stable `code` (`unknown_model`,
`invalid_datum`, `seq_conflict`, ...), the offending `param` when there is one, and the `request_id`.
`InsufficientBalanceError` (402, `insufficient_balance`) means your org's balance is used up: add credit in the
dashboard. A few codes have their own subclass: `LimitReachedError` (409 `limit_reached`) and its
`ProductionFullError` (409 `production_full`), `ModelNotInProductionError` (409 `model_not_in_production`) and
`ModelInProductionError` (409 `model_in_production`), all `ConflictError`s; and `NoPreviewError` (404 `no_preview`) and
`VersionNotCallableError` (404 `version_not_callable`), both `NotFoundError`s. A training operation that fails on the trainer raises `OperationFailedError` from `.result()`, with codes
such as `oom`, `no_gradients` or `trainer_lost`. Connection problems raise `APIConnectionError` and
`APITimeoutError`. A successful response whose body doesn't have the expected shape raises `ResponseValidationError`
(an `APIError`) naming the endpoint and the field, from every method.

Requests that fail with a connection error, a timeout, a 408, a 429 or a 5xx are retried with exponential backoff
and `Retry-After`, except `limit_reached`, `production_full` and `insufficient_balance`, which waiting doesn't
fix. Configure it with
`RetryPolicy`:

```python
client = endor.EndorClient(retry=endor.RetryPolicy(max_retries=5, timeout=120))
client = endor.EndorClient(retry=endor.RetryPolicy(max_retries=0))      # no retries
```

A model that hasn't been called for a while may start cold: its first decision can wait a few minutes. The SDK
waits up to 330 s for a decision (30 s for other calls). If the model is still not ready after 5 minutes, the API
answers `503 warming_up` with `Retry-After`, and the SDK retries after that wait.

Retries are safe: creates carry an `Idempotency-Key`, and a run's compute calls carry a sequence number the server
checks, so a retried request never runs twice.

`str(error)` ends with one line on how to report the problem (below); `error.message` is the server's message
alone, for showing to your own users.

## Reporting problems

If Endor does something you didn't expect, or the docs, an error message or the API are unclear, tell the Endor
team. This is meant for you and for the coding agent you handed the integration to: every error's `str()` ends with
the line below, so an agent sees it right where it gets stuck.

```python
endor.user_report("what you did, what happened, what you expected", kind="bug")       # no client needed
client.user_report("the docs don't say how to label a score question", kind="confusing")
client.user_report("the integration runs: tickets are routed with the tickets project", kind="works")
```

`kind` is `bug` (Endor did something unexpected), `confusing` (something is unclear) or `works` (send one once an
integration runs, so we know what worked). A report goes to `POST /v1/user_reports` with your message, the request it is
about (`request_id=`, by default the latest failed request of the client, or of any client in the process for
`endor.user_report`) and a small context: the SDK and Python versions, the platform, and that failure's status, code and
endpoint, plus anything you pass as `context={...}` (short values). Never your API key, a request body or a
decision's content. It needs your API key (the client's, or `ENDOR_API_KEY`). From a shell: `endor user-report "what happened" --kind bug [--request-id req_...]`.

If you give a coding agent the Endor integration, add to its instructions: *"If Endor behaves unexpectedly or a step
is unclear, report it with `endor.user_report(...)`; report `kind="works"` once the integration runs."*

## Configuration

| Argument | Environment variable | Default |
|---|---|---|
| `api_key` | `ENDOR_API_KEY` | required |
| `base_url` | `ENDOR_BASE_URL` | production |
| `model` | `ENDOR_DEFAULT_MODEL` | none: pass `model=` to `system_one`, e.g. `"tickets"` |
| `timeout` | | 30 s per request, 330 s for a decision (a future poll also waits up to 25 s on the server) |
| | `ENDOR_LOG_LEVEL` | unset; `info` logs one line per request on the `endor` logger |
| `report_latency` | `ENDOR_REPORT_LATENCY` | on; `False` or `0` stops sending `X-Endor-Client-Latency` (see below) |

`EndorClient` also takes `headers` for every request, `retry`, `gzip=False` to stop compressing large bodies, `capture=False`
to stop sending each run's settings and your git commit hash to the dashboard, and your own `httpx` client or
transport.

## What the SDK sends

Besides your API key and the request itself, every request carries:

- `User-Agent: endor-python/<version> (python <version>; <os>; <cpu>)`, with `; cli` added for the `endor` command;
- `X-Endor-SDK: endor-python`, `X-Endor-SDK-Version: <version>`, `X-Endor-Runtime: python/<version> (<os>; <cpu>)`
  and `X-Endor-SDK-Interface: python` or `cli`;
- `X-Endor-SDK-Method` (the SDK call, e.g. `run.forward_backward`), `X-Endor-SDK-Recipe` (inside a recipe),
  `X-Endor-Client-Request-Id` (the same on every retry of one call) and `X-Endor-Retry-Count` (on retries);
- `X-Endor-Client-Latency: <request id>=<ms>`: how long the previous request took from sending it to reading the
  response, network and connection set-up included, sent once with the next request (nothing waits for it), so the
  dashboard shows the latency you see next to Endor's own. `EndorClient(report_latency=False)` or
  `ENDOR_REPORT_LATENCY=0` turns it off.

`runs.create` also sends the run's settings and, unless `capture=False`, your git commit hash and a dirty flag
(never file contents). While a run is open, a background thread sends `POST /v1/runs/{id}/heartbeat`, with no
body, once a minute. A report (`endor.user_report`) sends only what "Reporting problems" lists.

For every request, the API stores the request id, your org, the calling key, the route, the status, the latency and
the end-to-end time the SDK reported for it, plus a `client` record with one field per header: `user_agent`, `sdk`, `sdk_version`, `runtime`, `interface`,
`method`, `recipe`, `client_request_id` and `retry_count` (each cut to 200 characters). It also keeps the request
body, and the response body when it is JSON and at most 256 KB, in object storage (S3). The secret API key is never
stored. You can add your own headers with `EndorClient(headers=...)`; the ones above can't be overridden.

## CLI

```
endor whoami
endor base-models
endor projects list | create NAME --base-model BASE | delete NAME
endor projects versions NAME | promote NAME VERSION | auto-promote NAME on|off
endor custom-models list [--production | --archived] | info NAME | promote NAME [--replace OTHER] | archive NAME
endor custom-models download NAME [-o DIR] [--include-optimizer] | set-ttl NAME SECONDS|none | delete NAME
endor datasets list | upload NAME FILE | delete NAME
endor runs list | show RUN_ID | close RUN_ID
endor eval MODEL DATASET
endor usage --start 2026-10-01 --end 2026-10-06 [--project P] [--csv]
endor user-report "what you did, what happened, what you expected" [--kind bug|confusing|works] [--request-id REQ]
```

Projects are continuous learning projects; custom models are the ones you train with the SDK. `MODEL` in `eval` is a
custom model's name or a base model's id (`owner/model`); `eval` prints the accuracy overall and per question. `endor base-models` prints
one line per base model: its Decision Index rank (`=` marks a tie), `tokens/s`, context (`max_question_tokens`), price
per 1M input tokens (called directly, or a custom model on it), and what it is meant for. `endor whoami` also
shows the org's balance (`balance_usd`), whether it has continuous learning (`continuous_learning`) and its limits. Every command takes `-f json` (or `--format json`), before or after the command (`endor -f json whoami` or `endor whoami -f json`), and needs `ENDOR_API_KEY`, `endor user-report` too. Training itself happens in Python. `endor runs close`
frees a slot when a script left a run open. `usage --csv` columns are `hour, kind, project, base_model, model,
training_run_id, input_tokens, gpu_seconds, continuous_learning, price_per_mtok, base_cost_usd,
continuous_learning_cost_usd, cost_usd`; `continuous_learning` is true for a continuous learning project's
decisions; a custom model's decisions have no project.

## Reference

| Object | Members |
|---|---|
| `EndorClient(*, api_key, model, retry, timeout, headers, base_url, capture, exclude_from_training, gzip, http_client, async_http_client, transport, async_transport, report_latency)` | `system_one(state, questions, *, model, retry, timeout, extra_headers, extra_body, exclude_from_training, response_model)`, `system_one_async`, `predict(model, rows, *, progress=True, timeout=None)` → one `{question: answer}` per row, `predict_async`, `models.list()`, `models.list_async()`, `base_models()`, `projects`, `custom_models`, `datasets`, `runs`, `evaluations`, `corrections`, `whoami()`, `balance()` → USD, `usage(starting_on, ending_before, project=None)`, `user_report(message, *, kind="bug", request_id=None, context=None)`, `user_report_async`, `close()`, `aclose()`, context managers |
| `endor.user_report(message, *, kind="bug", request_id=None, context=None, api_key=None, base_url=None, timeout=None)` | `endor.user_report_async(...)`; returns `UserReportReceipt` (`id`, `message`) |
| `client.projects` | `create(name, base_model, description=None)`, `get(name)`, `get_or_create(name, base_model, description=None)`, `list(limit=None, offset=0)` (all pages) |
| `Project` | `.name`, `.info_`, `.ref` (`"<project>"`), `.preview_ref` (`"<project>:preview"`, or None without a preview), `.production`, `.preview`, `.auto_promote`, `info()`, `learning()` → `Learning`, `versions()` → `list[VersionInfo]`, `promote(version)`, `set_auto_promote(on)`, `download(version, path, include_optimizer=False)`, `update(*, description)`, `delete()` |
| `client.corrections` | `create(request_id, questions, *, comment=None)` → `CorrectionInfo`, `batch(items)` → `list[CorrectionInfo]`, `get(request_id)`, `delete(request_id)`, each with an `_async` variant; `request_id` is a `SystemOneResponse` or its id |
| `client.custom_models` | `list(production=None, limit=None, offset=0)` → `list[CustomModelInfo]` (production first), `get(name)`, `create(name, base_model=None, *, run_name=None, **run_options)` → `Run`, `promote(name, replace=None)`, `archive(name)`, `set_ttl(name, ttl_seconds)`, `download(name, path, include_optimizer=False)`, `delete(name)` |
| `client.datasets` | `upload(name, rows)`, `list()`, `get(name)`, `rows(name, page=500)`, `delete(name)` |
| `client.runs` | `create(base_model=None, *, model, rank, alpha, seed, train_attn, train_mlp, train_readout, from_model, include_optimizer, name, tags, config, user_metadata, total_steps, wandb, wait, progress)` → `Run`, `get(run_id)`, `list(tag=None, limit=None, offset=0)` |
| `client.evaluations` | `create(model, dataset=None, run_id=None, *, rows=None)` → `APIFuture[Evaluation]`, `get(evaluation_id)`, `list(run_id=None, model=None)` |
| `Run` | `.id`, `.model`, `.info_`, `.ready`, `.next_seq_id`, `forward(data, loss_fn)`, `forward_backward(data, loss_fn)`, `optim_step(adam_params=None, *, learning_rate=None)`, `save_checkpoint(name=None, *, include_optimizer, ttl_seconds, user_metadata)` → `APIFuture[str]`, `predict(rows)` → `APIFuture[list[dict[str, Answer]]]`, the `_async` variants of those five, `close(*, wait=False, timeout=None)`, `close_async(...)`, `info()`, `set_total_steps(n)`, `log(metrics, step=None)`, `metrics(keys=None, since_step=None)`, `log_eval(model, results, *, step, name)`, sync and async context manager |
| `APIFuture[T]` | `result(timeout=None)`, `await f`, `result_async(timeout)`, `done()`, `info`, `cancel()`, `cancel_async()`, `APIFuture.completed(value)`; `endor.gather(*futures)`, `endor.gather_async(*futures)` |
| `endor.Steps(in_flight=2)` | `add(*futures)` → the results of the steps it read, `wait()` → the rest, `add_async`, `wait_async`, `len(steps)` |
| `endor.data` | `to_row`, `load_rows(path)`, `save_rows(path, rows)`, `label_target(question, label)`, `row_to_datums(row)`, `rows_to_datums(rows)`, `split(rows, holdout=0.1, seed=0)`, `batches(items, size, *, shuffle=True, seed=0)` |
| `endor.metrics` | `decision_metrics(probs, targets, bins=10)` → `{n, accuracy, nll, brier, ece, mean_confidence, selective}` |
| `endor.recipes.supervised` | `SupervisedConfig`, `SupervisedResult`, `train(cfg, rows, eval_rows=None, client=None)`, `evaluate_run(run, datums)`, `evaluate_model(client, model, rows)`, `lr_at(cfg, step, total)` |
| `endor.recipes.distill` | `Teacher` (protocol), `DistillConfig(supervised, teacher, budget, seed)`, `label_with_teacher(rows, cfg)`, `distill(cfg, unlabeled_rows, eval_rows, client=None)` |
| Types | `Noul`, `Choice`, `Score`, `NoulCriteria`; `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `SystemOneResponse`, `Usage`, `ModelMetadata`, `ListModelsResponse`, `BaseModelInfo`, `DecisionIndexInfo`; `Datum`, `DecisionRow`, `Target`, `LoraConfig`, `AdamParams`, `ForwardOutput`, `OptimStepOutput`; `CorrectionInfo`, `QuestionCorrectionInfo`, `QuestionCorrection`, `CorrectionAnswer`; `ProjectInfo`, `VersionInfo`, `Learning`, `Training`, `Drift`, `CustomModelInfo`, `RunInfo`, `LoraInfo`, `DatasetInfo`, `Evaluation`, `MetricPoint`, `DownloadedFile`, `UsageRow`, `WhoAmI`; helpers `option_keys`, `answer_probabilities`, `question_dict` |
| Errors | `EndorError`; `APIError` with `BadRequestError`, `AuthenticationError`, `InsufficientBalanceError`, `PermissionDeniedError`, `NotFoundError` (`NoPreviewError`, `VersionNotCallableError`), `ConflictError` (`LimitReachedError` > `ProductionFullError`, `ModelNotInProductionError`, `ModelInProductionError`), `PayloadTooLargeError`, `UnprocessableEntityError`, `RateLimitError`, `OverloadedError`, `InternalServerError`, `ResponseValidationError`; `APIConnectionError`, `APITimeoutError`; `OperationFailedError` |

Every public member has a docstring with the details.

## License

MIT
