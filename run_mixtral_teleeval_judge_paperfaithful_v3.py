import argparse
import json
import os
import re
from tqdm import tqdm
from llama_cpp import Llama, LlamaGrammar

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def parse_judgment(text: str):
    s = str(text).strip()
    s_up = re.sub(r'\s+', ' ', s.upper())

    if s_up == "YES":
        return 1, "YES", s
    if s_up == "NO":
        return 0, "NO", s
    if s_up == "":
        return None, "EMPTY", s
    return None, "OTHER", s

GRAMMAR = LlamaGrammar.from_string(r"""
root ::= answer
answer ::= "Yes" | "No"
""")

def judge_answer(llm, question, model_answer, ground_truth):
    prompt = f"""Task: Evaluate the correctness of a provided answer to a telecommunications and networking question.

Question: {question}
Ground Truth Answer: {ground_truth}
Provided Answer: {model_answer}

Instructions:
1. Compare the Provided Answer to the Ground Truth Answer.
2. Determine if the Provided Answer is correct based on the Ground Truth Answer.
3. Respond with only Yes or No.

Is the Provided Answer correct?"""

    out = llm(
        prompt,
        max_tokens=2,
        temperature=0,
        top_p=1,
        grammar=GRAMMAR,
    )

    raw = out["choices"][0]["text"]
    score, label, normalized = parse_judgment(raw)
    return score, label, raw, normalized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-gpu-layers", type=int, default=30)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    data = load_json(args.input)

    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=False,
    )

    results = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            results = json.load(f)
        print("RESUMING_FROM:", len(results))

    total = len(data)

    for qid, item in tqdm(data.items(), total=total, desc=os.path.basename(args.input)):
        if qid in results:
            continue

        question = item.get("question", "")
        model_answer = item.get("generated_answer", "")
        ground_truth = item.get("ground_truth", "")

        try:
            score, label, raw, normalized = judge_answer(llm, question, model_answer, ground_truth)
        except Exception as e:
            results[qid] = {
                "score": 0,
                "judge_label": "ERROR",
                "judge_raw": f"ERROR: {e}",
                "judge_normalized": "ERROR"
            }
            if len(results) % args.checkpoint_every == 0:
                save_json(results, args.output)
            continue

        results[qid] = {
            "score": 0 if score is None else score,
            "judge_label": label,
            "judge_raw": raw,
            "judge_normalized": normalized
        }

        if len(results) % args.checkpoint_every == 0:
            save_json(results, args.output)
            print(f"SAVED {len(results)}/{total}")

    save_json(results, args.output)

    correct = sum(v.get("score", 0) for v in results.values())
    acc = correct / len(results) if results else 0.0
    print(f"FINAL_COUNT: {len(results)}")
    print(f"FINAL_ACCURACY: {acc:.6f}")
    print(f"FINAL_ACCURACY_PERCENT: {acc*100:.3f}%")

if __name__ == "__main__":
    main()
