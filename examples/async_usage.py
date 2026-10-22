"""Async decisions and an async training step.

ENDOR_API_KEY=edk_... python examples/async_usage.py
"""

import asyncio

import endor
from endor import Noul


async def main() -> None:
    async with endor.EndorClient(model="tickets") as client:
        # Projects are synchronous; this one answers with its base model at once.
        client.projects.get_or_create("tickets", base_model="mapika/decider-2b")

        # Many decisions at once, through the project's production version.
        states = [f"ticket {i}: my invoice is wrong" for i in range(5)]
        results = await asyncio.gather(
            *(client.system_one_async(s, {"billing": Noul(instructions="Is this about billing?")}) for s in states)
        )
        for s, r in zip(states, results, strict=True):
            print(s, "->", round(r.nouls["billing"].noul, 3))

        # Advanced: a training step of a custom model. Calls on one run are serialized, so concurrent submits keep
        # their order. Creating the run is synchronous: provisioning blocks this thread for a minute or so.
        datums = endor.data.rows_to_datums(
            [{"state": s, "questions": {"b": Noul(instructions="Billing?")}, "labels": {"b": True}} for s in states]
        )
        # the run is closed on errors too
        async with client.custom_models.create("billing-check", base_model="mapika/decider-2b") as run:
            fb = await run.forward_backward_async(datums)
            opt = await run.optim_step_async(learning_rate=1e-4)
            out, step = await endor.gather_async(fb, opt)
            print("loss", out.loss, "step", step.step)


asyncio.run(main())
