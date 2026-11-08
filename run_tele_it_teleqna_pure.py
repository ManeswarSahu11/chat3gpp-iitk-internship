import json
import os
from tqdm import tqdm
from server.llm import load_llama3_model, generate_answer_llama3

RESULTS_FILE = "outputs/teleqna/tele_it_teleqna_pure.json"
DATA_FILE = "data/teleqna/teleqna_paper.json"

def build_options_text(q_info):
    options = []
    for i in range(1, 6):
        key = f"option {i}"
        if key in q_info:
            options.append(f"{i}. {q_info[key]}")
    return "\n".join(options)

def extract_answer_number(text: str):
    import re
    text = str(text).strip()
    m = re.search(r'\b([1-5])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'Answer\s*[:\-]?\s*([1-5])', text, flags=re.I)
    if m:
        return m.group(1)
    return None

def extract_gold_option(x):
    import re
    x = str(x).strip()
    m = re.search(r'option\s*([1-5])', x, flags=re.I)
    if m:
        return m.group(1)
    return None

def main():
    os.makedirs("outputs/teleqna", exist_ok=True)

    model, tokenizer = load_llama3_model(
        model_name="LLama-3-8B-Tele-it",
        device="cuda:0"
    )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = {}

    for qid, q_info in tqdm(dataset.items(), total=len(dataset)):
        question = q_info["question"]
        options_text = build_options_text(q_info)
        correct_answer = str(q_info.get("answer", "")).strip()
        gold_option = extract_gold_option(correct_answer)

        prompt = f"""You are given a multiple-choice telecommunications question.

Question:
{question}

Options:
{options_text}

Answer with only the option number.
"""

        generated = generate_answer_llama3(model, tokenizer, prompt)
        pred = extract_answer_number(generated)

        results[qid] = {
            "generated_answer": generated,
            "predicted_answer": pred,
            "gold_option": gold_option,
            "is_correct": str(pred) == str(gold_option),
        }

        if len(results) % 10 == 0:
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    correct = sum(1 for v in results.values() if v["is_correct"])
    acc = correct / total * 100

    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", f"{acc:.3f}%")

if __name__ == "__main__":
    main()
