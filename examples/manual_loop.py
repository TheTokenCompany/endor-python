"""Train a custom model with your own loop: batches, losses, learning rate, evaluation and checkpoints are all in your
hands. Then promote it, so decisions can call it by its name.

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

if "heldout" not in {d.name for d in client.datasets.list()}:
    client.datasets.upload("heldout", heldout_rows)

with client.custom_models.create("router-2", base_model="decider-2b", rank=16, tags=["manual"]) as run:
    steps = endor.Steps()  # the next step is sent before the last one's results are read: 2 steps in flight
    sent = 0
    for epoch in range(2):
        for batch in endor.data.batches(train, 16, seed=epoch):
            fb = run.forward_backward(batch)  # gradients accumulate on the trainer
            opt = run.optim_step(learning_rate=1e-4)  # AdamW step, then zero gradients
            sent += 1
            for out, step in steps.add(fb, opt):  # the step before this one, once two are out
                print(f"step {step.step}: loss {out.loss:.3f}")
            if sent % 10 == 0:
                scored = run.forward(heldout).result()  # runs right after step `sent`: the current adapter
                metrics = decision_metrics(scored.probabilities, [d.target for d in heldout])
                run.log({"heldout/accuracy": metrics["accuracy"], "heldout/ece": metrics["ece"]}, step=sent)
                print(f"step {sent}: held-out accuracy {metrics['accuracy']:.3f}")
    for out, step in steps.wait():  # the last step
        print(f"step {step.step}: loss {out.loss:.3f}")

    model = run.save_checkpoint(include_optimizer=True).result()  # "router-2", saved inactive

print("saved", model)
evaluation = client.evaluations.create(model, "heldout", run_id=run.id).result()  # server-side scoring, not live yet
print(evaluation.results)
print(client.evaluations.create("decider-2b", "heldout").result().results)  # the untuned base

client.custom_models.promote(model)  # live: client.system_one(..., model="router-2") answers now

# Resume exactly where that run stopped (a new run, so a new GPU; it saves under a new name; close it when done):
with client.custom_models.create("router-3", from_model=model, include_optimizer=True) as resumed:
    print("resumed as", resumed.id, "at step", resumed.info_.step)
