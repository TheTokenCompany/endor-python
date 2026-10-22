"""Train a custom model on labeled rows in a few lines, promote it, then decide with it.

    ENDOR_API_KEY=edk_... python examples/fine_tune.py rows.jsonl

``rows.jsonl`` holds decision rows (README, "Data format"): a state, named questions and the labels you know.
"""

import sys

import endor
from endor.recipes import SupervisedConfig, supervised

rows = endor.data.load_rows(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl")

cfg = SupervisedConfig(model_name="router-1", base_model="mapika/decider-2b", eval_every=10)
result = supervised.train(cfg, rows)

print("model:", result.model)
if result.base_metrics:
    print("held-out accuracy, base:", round(result.base_metrics["accuracy"], 3))
print("held-out accuracy, tuned:", round(result.final_metrics["accuracy"], 3))

client = endor.EndorClient()
client.custom_models.promote(result.model)  # in production now, so decisions can call it
row = rows[0]
res = client.system_one(row.state, row.questions, model=result.model)  # "router-1"
print(res.answers)
