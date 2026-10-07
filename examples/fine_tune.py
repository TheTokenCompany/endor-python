"""Fine-tune a model on labeled rows in a few lines, then decide with it.

    ENDOR_API_KEY=edk_... python examples/fine_tune.py rows.jsonl

``rows.jsonl`` holds decision rows (README, "Data format"): a state, named questions and the labels you know.
"""

import sys

import endor
from endor.recipes import SupervisedConfig, supervised

rows = endor.data.load_rows(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl")

cfg = SupervisedConfig(project="tickets", model_name="v1", base_model="pplx-decider-v1.1-27b", eval_every=10)
result = supervised.train(cfg, rows)

print("model:", result.model)
if result.base_metrics:
    print("held-out accuracy, base:", round(result.base_metrics["accuracy"], 3))
print("held-out accuracy, tuned:", round(result.final_metrics["accuracy"], 3))

client = endor.EndorClient()
row = rows[0]
res = client.system_one(row.state, row.questions, model=result.model)
print(res.answers)
