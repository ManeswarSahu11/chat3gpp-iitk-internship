import json
from server.llm import load_llama3_model, generate_answer_llama3

with open("data/teleqna/teleqna_paper.json", "r", encoding="utf-8") as f:
    data = json.load(f)

qid, q = next(iter(data.items()))
question = q["question"]

options = []
for i in range(1, 6):
    key = f"option {i}"
    if key in q:
        options.append(f"{i}. {q[key]}")

prompt = f"""You are given a multiple-choice telecommunications question.

Question:
{question}

Options:
{chr(10).join(options)}

Answer with only the option number.
"""

print("QID:", qid)
print("PROMPT:")
print(prompt)
print("=" * 80)

model, tokenizer = load_llama3_model(
    model_name="LLama-3-8B-Tele-it",
    device="cuda:0"
)

out = generate_answer_llama3(model, tokenizer, prompt)

print("MODEL OUTPUT:")
print(repr(out))
