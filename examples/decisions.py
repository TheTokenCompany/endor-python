"""Ask a continuous learning project named questions about a state. "tickets" answers with the project's production
version (its base model until you promote one). The project learns from its decisions and from your corrections: new
versions wait in "tickets:preview" until you promote one, or go to production by themselves with auto-promote on.

ENDOR_API_KEY=edk_... python examples/decisions.py
"""

from endor import Choice, ChoiceAnswer, EndorClient, Noul, NoulAnswer, Score, SystemOneResponse

client = EndorClient(model="tickets")  # the default model for every call below
project = client.projects.get_or_create("tickets", base_model="decider-2b")

state = {"subject": "Charged twice", "body": "I was billed two times for my March invoice. Please fix this today."}
questions = {
    "department": Choice(
        instructions="Which team should handle `body`?",
        criteria={"billing": "Payments, invoices, refunds", "technical": "Bugs, outages", "sales": None},
    ),
    "urgent": Noul(instructions="Does the customer need an answer today?"),
    "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"]),
}

res = client.system_one(state, questions)  # the client's default model, "tickets"
print(res.model)  # what answered: "tickets:v4", or "tickets" while its base model answers
print(res.choices["department"].choice, res.choices["department"].probabilities)
print(res.nouls["urgent"].noul)
print(res.scores["frustration"].score, res.scores["frustration"].probabilities)

# It was wrong about one question: tell the project, and it learns from it.
client.corrections.create(res, {"urgent": {"correct": False}})


# A typed response: one attribute per question.
class Triage(SystemOneResponse):
    department: ChoiceAnswer
    urgent: NoulAnswer


triage = client.system_one(state, questions, response_model=Triage)
print(triage.department.choice, triage.urgent.noul)

# A new version waits as the preview: try it, then promote it (or project.set_auto_promote(True) for every new one).
info = project.info()
if info.preview:
    print(client.system_one(state, questions, model=project.preview_ref).model)  # "tickets:preview" answers
    project.promote(info.preview)  # in production: "tickets" answers with it now

# Every name a decision can send: your projects, their previews (when they have one) and your live custom models.
for m in client.models.list().models:
    print(m.name, m.kind)
