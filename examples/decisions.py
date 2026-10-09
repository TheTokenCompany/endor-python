"""Ask a model named questions about a state. Every decision goes through a project: "tickets/v1" is a saved model,
"tickets/decider-2b" one of its base models, and "tickets" a managed project's newest version (a custom project's
first base model).

ENDOR_API_KEY=edk_... python examples/decisions.py
"""

from endor import Choice, ChoiceAnswer, EndorClient, Noul, NoulAnswer, Score, SystemOneResponse

client = EndorClient(model="tickets")  # the default model for every call below
client.projects.get_or_create("tickets", base_models=["decider-2b"])

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
print(res.model)  # the model that answered: "tickets/decider-2b" in a custom project
print(res.choices["department"].choice, res.choices["department"].probabilities)
print(res.nouls["urgent"].noul)
print(res.scores["frustration"].score, res.scores["frustration"].probabilities)


# A typed response: one attribute per question.
class Triage(SystemOneResponse):
    department: ChoiceAnswer
    urgent: NoulAnswer


triage = client.system_one(state, questions, model="tickets/decider-2b", response_model=Triage)
print(triage.department.choice, triage.urgent.noul)

# Every name you can call: each project, its base models and your saved models.
for m in client.models.list().models:
    print(m.name, m.kind)
