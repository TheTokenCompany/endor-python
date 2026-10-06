# Decision data format

Endor trains and evaluates on **decision rows**. Each row is a decision request (`state` plus named `questions`)
with a label for every question you know the answer to. If you have called `system_one`, a row is that request
with the answers added.

Rows are JSON objects. Upload them to a project with `project.datasets.upload(name, rows)` (up to 50,000 rows per
dataset), or pass them straight to the training helpers in `endor.data` and `endor.recipes`. A project's datasets are
shared by all its runs and evaluations.

## A row

```json
{
  "id": "ticket-0001",
  "state": {"subject": "Charged twice", "body": "I was billed two times for my March invoice."},
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle `body`?",
                    "criteria": {"billing": "Payments, invoices, refunds", "technical": "Bugs, outages", "sales": null}},
    "urgent":      {"type": "noul", "instructions": "Does the customer need an answer today?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Annoyed", "Angry"]}
  },
  "labels": {"department": "billing", "urgent": true, "frustration": 1},
  "weight": 1.0
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | no | Your id for the row. Shown on the dashboard and in evaluation results. |
| `state` | yes | The data the decision is about: a string, a JSON object or a JSON array. Text only. |
| `questions` | yes | Named questions, each answered independently against `state`. The same objects `system_one` takes. |
| `labels` | no | The correct answer per question name. Questions without a label are unlabeled. |
| `weight` | no | How much the row counts in training (default 1). |

## Question types

| `type` | `criteria` | Limits |
|---|---|---|
| `noul` (yes/no) | optional `{"true": ..., "false": ...}` describing each outcome | |
| `choice` | required map `option_key → description` (a description may be `null`) | 1 to 255 options; some base models read fewer, see `client.base_models()` |
| `score` | required ordered list of level descriptions, lowest first | 2 to 10 levels |

`instructions`, option descriptions and score levels can be plain text or JSON. Refer to parts of the state with
backticked paths: ``"Does `ticket.body` ask for a refund?"``.

## Option ids

Every option of a question has an id, and the SDK uses these ids everywhere: in labels, in training targets and in
answer probabilities.

| Type | Option ids |
|---|---|
| `noul` | `"false"`, `"true"` |
| `choice` | the `criteria` keys, in order |
| `score` | `"0"` … `"n-1"`, the level index |

## Labels

| Question type | Hard label | Soft label (a probability distribution, for example from a teacher model) |
|---|---|---|
| `noul` | `true` or `false` | `0.8` (P(true)), or `{"true": 0.8, "false": 0.2}` |
| `choice` | `"billing"` (an option key) | `{"billing": 0.7, "technical": 0.2, "sales": 0.1}` (every option) |
| `score` | `2` (level index, 0 = first level) | `[0.1, 0.6, 0.3]` (one probability per level), or `{"0": 0.1, "1": 0.6, "2": 0.3}` |

Soft labels must cover every option, be non-negative and sum to 1 (±0.02).

## How rows are used

- **Training.** Each labeled question becomes one training example (state + that question + its label), weighted by
  `weight`. Unlabeled questions are skipped. `endor.data.rows_to_datums(rows)` does this conversion.
- **Evaluation.** The model answers every labeled question of a held-out dataset. Reported metrics: accuracy, log
  loss, Brier score, calibration (ECE) and accuracy at confidence thresholds.
- **Teacher distillation.** Unlabeled questions can be labeled by a teacher first (`endor.recipes.distill`). The
  teacher's answer distribution becomes a soft label.

## Other row shapes

`endor.data.to_row`, `load_rows` and `datasets.upload` also accept two single-question shapes. Each becomes a row
with one question named `decision`.

- `{"state", "question", "expected"}`: `expected` is the label, as `"yes"`/`"no"` (or a bool) for noul, an option
  key for choice, or a level index for score.
- `{"state", "question", "label"?, "target"?}`: `label` is a hard label; a list, float or object `target` is a
  soft label.

## Files

`endor.data.load_rows(path)` reads a `.jsonl` file (one row per line) or a `.json` file (an array of rows).
`endor.data.save_rows(path, rows)` writes `.jsonl` in the native shape.
