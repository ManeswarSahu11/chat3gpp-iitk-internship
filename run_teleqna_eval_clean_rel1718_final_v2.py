import json
import os
import re
import warnings
from tqdm import tqdm
from elasticsearch import Elasticsearch
from retrievers.VectorRetriever import VectorRetrieval
from configs.model_configs import EMBED_CONFIG, LLM_CONFIG
from server.embedding import load_embeddings
from server.llm import load_llama3_model, generate_answer_llama3

warnings.filterwarnings("ignore")
VectorRetrieval.__abstractmethods__ = frozenset()

RESULTS_FILE = "outputs/teleqna/teleqna_results_clean_rel1718_final_v2.json"
EVAL_DATA_FILE = "data/teleqna/teleqna_paper.json"
INDEX_NAME = "index_clean_rel1718_final_v2"
CHECKPOINT_EVERY = 10

MCQ_PROMPT = """You are an expert on 3GPP standards.

Based on the provided context, answer the multiple-choice question.

Context:
{context}

Question:
{question}

Options:
{options}

Instructions:
- Select ONLY ONE correct option.
- Use ONLY the context.
- If unsure, say "Insufficient context to answer."

Output format:
"answer": "option X: [full option text]"

Answer:
"""

def search_docs(query, embed_model, es_client):
    kb = VectorRetrieval(embed_model, es_client=es_client)
    docs, _ = kb.search_rrf(INDEX_NAME, query, top_k1=1000, top_k2=5)
    return docs

def main():
    os.makedirs("outputs/teleqna", exist_ok=True)

    es_client = Elasticsearch(
        "http://localhost:9200",
        request_timeout=120,
        max_retries=3,
        retry_on_timeout=True
    )
    if not es_client.ping():
        print("Could not connect to Elasticsearch.")
        raise SystemExit(1)

    embed_model = load_embeddings(
        model=EMBED_CONFIG["embed_model"],
        device=EMBED_CONFIG["embed_device"]
    )
    llama3_model, llama3_tokenizer = load_llama3_model(
        model_name=LLM_CONFIG["llm_model"],
        device=LLM_CONFIG["llm_device"]
    )

    with open(EVAL_DATA_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)

    correct = sum(1 for v in results.values() if v.get("is_correct", False))

    for q_key, q_info in tqdm(eval_data.items(), total=len(eval_data)):
        if q_key in results:
            continue

        question = q_info.get("question", "")
        correct_answer = q_info.get("answer", "")
        options_text = "\n".join(
            f"{k}: {v}" for k, v in q_info.items() if k.startswith("option")
        )

        try:
            retrieved_docs = search_docs(question, embed_model, es_client)
            context_text = "\n\n".join(retrieved_docs)

            prompt = MCQ_PROMPT.format(
                context=context_text,
                question=question,
                options=options_text
            )

            answer_text = generate_answer_llama3(
                llama3_model,
                llama3_tokenizer,
                prompt
            )

            pred_match = re.search(r'option\s*(\d+)', answer_text, re.IGNORECASE)
            predicted = pred_match.group(1) if pred_match else None

            gold_match = re.search(r'option\s*(\d+)', correct_answer, re.IGNORECASE)
            correct_num = gold_match.group(1) if gold_match else None

            is_correct = predicted == correct_num
            if is_correct:
                correct += 1

            results[q_key] = {
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": answer_text,
                "predicted": predicted,
                "is_correct": is_correct
            }

            if len(results) % CHECKPOINT_EVERY == 0:
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            print(f"\nError on {q_key}: {e}")
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print(f"\nTotal: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}%")

if __name__ == "__main__":
    main()
