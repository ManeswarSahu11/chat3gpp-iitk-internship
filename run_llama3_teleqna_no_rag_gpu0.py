import json
import os
import re
from tqdm import tqdm
from server.llm import load_llama3_model, generate_answer_llama3

RESULTS_FILE = "outputs/teleqna/llama3_teleqna_no_rag.json"
DATA_FILE = "data/teleqna/teleqna_paper.json"

def build_options_text(q_info):
    options = []
    for i in range(1, 6):
        key = f"option {i}"
        if key in q_info:
            options.append(f"{i}. {q_info[key]}")
    return "\n".join(options)

def extract_answer_number(text):
    if text is None:
        return None
    text = str(text)
    m = re.search(r'\b([1-5])\b', text)
    return m.group(1) if m else None

def extract_gold_option(x):
    x = str(x)
    m = re.search(r'option\s*([1-5])', x, flags=re.I)
    return m.group(1) if m else None

def main():
    os.makedirs("outputs/teleqna", exist_ok=True)

    model, tokenizer = load_llama3_model(
        model_name="Llama-3-8B-Instruct",
        device="cuda:0"
    )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = {}

    for qid, q_info in tqdm(dataset.items()):
        question = q_info["question"]
        options_text = build_options_text(q_info)
        correct_answer = q_info.get("answer")

        prompt = f"""You are given a multiple-choice telecommunications question.

Question:
{question}

Options:
{options_text}

Answer with only the option number.
"""

        generated = generate_answer_llama3(model, tokenizer, prompt)
        pred = extract_answer_number(generated)
        gold = extract_gold_option(correct_answer)

        results[qid] = {
            "generated_answer": generated,
            "predicted_answer": pred,
            "gold_option": gold,
            "is_correct": pred == gold
        }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    correct = sum(1 for v in results.values() if v["is_correct"])
    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", f"{(correct/total*100):.3f}%")

if __name__ == "__main__":
    main()
