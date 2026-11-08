import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = "data/teleeval_58627_exact.json"
MODEL_PATH = "/scratch/work/yash22/models/teleit_8b_clean"
OUT_PATH = "outputs/teleeval/tele_it_teleeval_no_rag.json"

def build_prompt(question: str) -> str:
    return f"""The following is a question about telecommunications and networking. Just give the answer.

Question:{question}
Answer:"""

def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        return list(d.items())
    if isinstance(d, list):
        return [(str(i), x) for i, x in enumerate(d)]
    raise ValueError(type(d))

def get_question(sample):
    for k in ["question", "statement", "query", "prompt"]:
        if isinstance(sample, dict) and k in sample and sample[k]:
            return str(sample[k]).strip()
    raise KeyError(f"Question field not found in sample keys: {list(sample.keys())}")

def get_gt(sample):
    if not isinstance(sample, dict):
        return None
    for k in ["ground_truth", "answer", "reference_answer", "gold"]:
        if k in sample:
            return sample[k]
    return None

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    items = load_items(DATA_PATH)
    print("TOTAL_ITEMS:", len(items))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        print("RESUMING_FROM:", len(results))

    for qid, sample in tqdm(items, desc="Generating"):
        if qid in results:
            continue

        question = get_question(sample)
        gt = get_gt(sample)
        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

        results[qid] = {
            "id": qid,
            "question": question,
            "ground_truth": gt,
            "generated_answer": answer,
            "model": "Llama-3-8B-Tele-it",
            "rag": False
        }

        if len(results) % 100 == 0:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("SAVED_TO:", OUT_PATH)
    print("TOTAL_SAVED:", len(results))

if __name__ == "__main__":
    main()
