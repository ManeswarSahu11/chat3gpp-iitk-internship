import json
import os
from tqdm import tqdm
from elasticsearch import Elasticsearch
from server.embedding import load_embeddings
from server.llm import load_llama3_model, generate_answer_llama3
from retrievers.VectorRetriever import VectorRetrieval

INDEX_NAME = "index_clean_rel1718_final"
RESULTS_FILE = "outputs/teleeval/tele_it_teleeval_results_clean_rel1718_final.json"
EVAL_DATA_FILE = "data/teleeval_58627_exact.json"

VectorRetrieval.__abstractmethods__ = frozenset()

def search_docs(question, embed_model, es_client):
    kb = VectorRetrieval(embed_model=embed_model, es_client=es_client)
    docs, _ = kb.search_rrf(INDEX_NAME, question, top_k1=1000, top_k2=5)
    return docs

def main():
    os.makedirs("outputs/teleeval", exist_ok=True)

    es_client = Elasticsearch(
        "http://localhost:9200",
        request_timeout=120,
        max_retries=3,
        retry_on_timeout=True
    )

    embed_model = load_embeddings(model="bge-m3", device="cuda:1")
    model, tokenizer = load_llama3_model(
        model_name="LLama-3-8B-Tele-it",
        device="cuda:1"
    )

    with open(EVAL_DATA_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)

    for qid, q_info in tqdm(dataset.items(), total=len(dataset)):
        if qid in results:
            continue

        try:
            question = q_info["question"]
            docs = search_docs(question, embed_model, es_client)
            context = "\n\n".join(docs)

            prompt = f"""You are a telecommunications expert. Answer the question using the provided context.

Context:
{context}

Question:
{question}

Answer concisely and accurately.
"""

            generated = generate_answer_llama3(model, tokenizer, prompt)

            results[qid] = {
                "question": question,
                "generated_answer": generated,
            }

            if len(results) % 10 == 0:
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error on question {qid}: {e}")
            continue

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Total generated:", len(results))

if __name__ == "__main__":
    main()
