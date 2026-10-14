---
name: endor
license: MIT
description: >
  Set up and build with Endor (endor.thetokencompany.com): fine-tune open decision
  models on your own labeled data with the `endor` Python SDK, measure them against
  the base model, and call them through the TypeSafe-compatible decision API
  (noul, choice and score questions with probabilities). Use when a project uses
  Endor or the `endor` package, when setting up an Endor API key, or when a
  classify, route, score or yes/no step should become a model trained on the
  project's own decisions.
---

# Build with Endor

Endor fine-tunes open decision models on your data and serves them behind one API call. A decision model reads a
**state** (text, a JSON object or images) and answers typed **questions** with probabilities: `noul` (yes or no),
`choice` (one of a set) and `score` (a level on one scale). The decision API is TypeSafe's System One API, so TypeSafe
clients work with Endor too.

## Read the live docs

**The live docs are the source of truth. Read them as part of the task**, and don't invent fields, ids or prices.

- Start with the index, https://endor.thetokencompany.com/llms.txt, then read the pages the task needs. Every docs
  page has a Markdown version: add `.md` to its URL.
- The API schema: https://api.endor.thetokencompany.com/v1/openapi.json
- If the docs can't be fetched, use the installed SDK (typed, with docstrings) and say so.

| Task | Read |
|---|---|
| First setup, end to end | [Quickstart](https://endor.thetokencompany.com/docs/quickstart.md) |
| Call a model from an app | [Decisions](https://endor.thetokencompany.com/docs/decisions.md) |
| Prepare labeled data | [Data format](https://endor.thetokencompany.com/docs/data-format.md) |
| Write a training loop | [Training](https://endor.thetokencompany.com/docs/training.md) |
| Pick a base model; prices, limits, cold starts | [Base models](https://endor.thetokencompany.com/docs/base-models.md) |
| Every SDK call, the CLI | [Python SDK](https://endor.thetokencompany.com/docs/sdk.md) |
| Every HTTP endpoint and error | [API reference](https://endor.thetokencompany.com/docs/api.md) |

## Set up

1. Install the SDK with the project's package manager: `pip install endor` or `uv add endor` (Python 3.10+).
2. The API key (`edk_...`, from **API keys** in the dashboard) goes in `ENDOR_API_KEY`: the environment, or a `.env`
   that is in `.gitignore`. `endor.EndorClient()` and the `endor` CLI read it. If it is missing, ask the user to add
   it there. Never ask them to paste the key into the chat, and never print it, commit it or write it into a command.
   In a web app, keep it on the server.
3. `ENDOR_BASE_URL` is needed only when the user's dashboard is not https://endor.thetokencompany.com: the SDK calls
   https://api.endor.thetokencompany.com by default.
4. `endor whoami` checks the key and shows the organization and its limits.
5. A new organization has $0. Evaluations, training and decisions return `402 insufficient_balance` until the user adds
   credits under **Billing** in the dashboard. Ask them to; don't work around it.

## How Endor fits together

- An organization has projects. A project is `custom` (you train with the SDK) or `managed` (Endor trains from its
  decisions; coming soon), fixed at creation. A project has one or more base models.
- A decision always names a project: `<project>/<name>` (a saved model), `<project>/<base id>` (one of the project's
  base models) or `<project>`. A bare base model id fails with `422 model_requires_project`.
- A base model idle for 15 minutes starts cold: the first decision waits 1 to 3 minutes. The SDK waits; over HTTP, use
  a 330 s timeout and retry `503 warming_up` after its `Retry-After`.
- A run holds a GPU. Open it with `with project.runs.create() as run:` so it closes, also after an error.

## Fine-tune a model (custom project)

Follow the [Quickstart](https://endor.thetokencompany.com/docs/quickstart.md): upload labeled rows, measure the base
model on held-out rows, train, measure again, and use the better model. `endor.recipes.supervised` does all of it in
one call:

```python
import endor
from endor.recipes.supervised import SupervisedConfig, train

rows = endor.data.load_rows("train.jsonl")  # one labeled decision row per line
cfg = SupervisedConfig(project="tickets", model_name="v1", epochs=3, batch_size=8, warmup_steps=2, holdout=0.2)
result = train(cfg, rows)  # creates the project, measures tickets/decider-2b, trains, measures, saves tickets/v1
print(result.base_metrics["accuracy"], "->", result.final_metrics["accuracy"])
```

Without data of their own, the user can start from the sample at
https://endor.thetokencompany.com/docs/samples/train.jsonl. Always compare with the base model on rows the model was
not trained on before calling a model better.

## Call a model

```python
import endor
from endor import Choice, Noul

client = endor.EndorClient()
res = client.system_one(
    {"body": "I was billed two times for my March invoice."},
    {"department": Choice(instructions="Which team must handle `body`?",
                          criteria={"billing": "Payments, refunds", "technical": "Bugs, outages"}),
     "urgent": Noul(instructions="Does the customer need an answer today?")},
    model="tickets/v1",
)
print(res.choices["department"].choice, res.nouls["urgent"].noul)
```

Put the questions in one place in the code, so the user can review them. Wrap the call in a small function with a
timeout and error handling, and use it where the app makes this decision.

## Write good questions

- One narrow judgment per question. Ask independent questions about the same state in one call.
- Give the question the state it needs; with several parts, use named JSON fields and refer to them in backticks,
  like `` `body` ``.
- `choice`: `criteria` is an object, option id to description (or `null`). Add a "none of these" option when nothing
  may fit.
- `score`: `criteria` is a list of the levels of one scale, lowest first, 2 to 10 levels; not a list of things to
  check.
- `noul`: the answer is the probability of yes. Pick thresholds from the user's data, not by guessing.

## When something goes wrong

- Errors are `{"error": {"code", "message", "param", "request_id", "report"}}` with stable codes; the SDK raises a
  typed `EndorError` subclass. Look the code up in the docs before changing code.
- If Endor does something you don't expect, or the docs or an error are unclear, tell the Endor team:
  `endor.user_report("what you did, what happened, what you expected", kind="bug")` (or `kind="confusing"`), or
  `endor user-report "..." --kind bug`. When an integration works, report `kind="works"` once. Every report is read.
