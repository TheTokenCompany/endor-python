"""Ask a model named questions about a state.

ENDOR_API_KEY=edk_... python examples/decisions.py
"""

from endor import Choice, ChoiceAnswer, EndorClient, Noul, NoulAnswer, Score, SystemOneResponse

client = EndorClient()

state = {"subject": "Charged twice", "body": "I was billed two times for my March invoice. Please fix this today."}
questions = {
    "department": Choice(
        instructions="Which team should handle `body`?",
        criteria={"billing": "Payments, invoices, refunds", "technical": "Bugs, outages", "sales": None},
    ),
    "urgent": Noul(instructions="Does the customer need an answer today?"),
    "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"]),
}

res = client.system_one(state, questions)  # the client's default model
print(res.model)
print(res.choices["department"].choice, res.choices["department"].probabilities)
print(res.nouls["urgent"].noul)
print(res.scores["frustration"].score, res.scores["frustration"].probabilities)


# A typed response: one attribute per question.
class Triage(SystemOneResponse):
    department: ChoiceAnswer
    urgent: NoulAnswer


triage = client.system_one(state, questions, response_model=Triage)
print(triage.department.choice, triage.urgent.noul)

# The catalog: base models and your saved models.
for m in client.models.list().models:
    print(m.name, m.kind)
