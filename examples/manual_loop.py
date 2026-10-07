"""Drive the training loop yourself: batches, losses, learning rate, evaluation and checkpoints are all in your hands.

ENDOR_API_KEY=edk_... python examples/manual_loop.py rows.jsonl
"""

import logging
import sys

import endor
from endor.metrics import decision_metrics

logging.basicConfig(level=logging.INFO)  # shows provisioning progress and one line per request

client = endor.EndorClient()
rows = endor.data.load_rows(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl")
train_rows, heldout_rows = endor.data.split(rows, holdout=0.1, seed=0)
train = endor.data.rows_to_datums(train_rows)
heldout = endor.data.rows_to_datums(heldout_rows)

project = client.projects.get_or_create("tickets")
if "heldout" not in {d.name for d in project.datasets.list()}:
    project.datasets.upload("heldout", heldout_rows)

with project.runs.create(base_model="pplx-decider-v1.1-27b", rank=16, tags=["manual"]) as run:
    steps = 0
    for epoch in range(2):
        for batch in endor.data.batches(train, 16, seed=epoch):
            fb = run.forward_backward(batch)  # gradients accumulate on the trainer
            opt = run.optim_step(learning_rate=1e-4)  # AdamW step, then zero gradients
            out, step = endor.gather(fb, opt)  # one wait for both
            steps += 1
            if steps % 10 == 0:
                scored = run.forward(heldout).result()  # the current adapter, no checkpoint needed
                metrics = decision_metrics(scored.probabilities, [d.target for d in heldout])
                run.log({"heldout/accuracy": metrics["accuracy"], "heldout/ece": metrics["ece"]})  # at the current step
                print(f"step {step.step}: loss {out.loss:.3f}  held-out accuracy {metrics['accuracy']:.3f}")

    model = run.save_checkpoint("v2", include_optimizer=True).result()  # "tickets/v2"

print("saved", model)
evaluation = project.evaluate(model, "heldout", run_id=run.id).result()  # server-side scoring
print(evaluation.results)

# Resume exactly where that run stopped (a new run, so a new GPU; close it when done):
with project.runs.create(from_model="v2", include_optimizer=True) as resumed:
    print("resumed as", resumed.id, "at step", resumed.info_.step)
