"""Async decisions and an async training step.

ENDOR_API_KEY=edk_... python examples/async_usage.py
"""

import asyncio

import endor
from endor import Noul


async def main() -> None:
    async with endor.EndorClient() as client:
        # Many decisions at once.
        states = [f"ticket {i}: my invoice is wrong" for i in range(5)]
        results = await asyncio.gather(
            *(client.system_one_async(s, {"billing": Noul(instructions="Is this about billing?")}) for s in states)
        )
        for s, r in zip(states, results, strict=True):
            print(s, "->", round(r.nouls["billing"].noul, 3))

        # A training step. Calls on one run are serialized, so concurrent submits keep their order.
        # Projects and runs.create are synchronous: provisioning blocks this thread for a few minutes.
        project = client.projects.get_or_create("tickets")
        datums = endor.data.rows_to_datums(
            [{"state": s, "questions": {"b": Noul(instructions="Billing?")}, "labels": {"b": True}} for s in states]
        )
        async with project.runs.create(base_model="pplx-decider-v1-27b") as run:  # closed even on errors
            fb = await run.forward_backward_async(datums)
            opt = await run.optim_step_async(learning_rate=1e-4)
            out, step = await endor.gather_async(fb, opt)
            print("loss", out.loss, "step", step.step)


asyncio.run(main())
