
import json
import os
import warnings
from tqdm import tqdm
from elasticsearch import Elasticsearch
from retrievers.VectorRetriever import VectorRetrieval
from configs.model_configs import EMBED_CONFIG, LLM_CONFIG
from server.embedding import load_embeddings
from server.llm import load_llama3_model, generate_answer_llama3

warnings.filterwarnings("ignore")
VectorRetrieval.__abstractmethods__ = frozenset()

RESULTS_FILE = "outputs/teleeval/chat3gpp_teleeval_results_clean_rel1718_final_v2.json"
EVAL_DATA_FILE = "data/teleeval_58627_exact.json"
INDEX_NAME = "index_clean_rel1718_final_v2"
CHECKPOINT_EVERY = 10

OPEN_PROMPT = """The following is a question about telecommunications and networking.
Just give the answer based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

def search_docs(query, embed_model, es_client):
    kb = VectorRetrieval(embed_model, es_client=es_client)
    docs, _ = kb.search_rrf(INDEX_NAME, query, top_k1=1000, top_k2=5)
    return docs

def main():
    os.makedirs("outputs/teleeval", exist_ok=True)

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

    for q_key, q_info in tqdm(eval_data.items(), total=len(eval_data)):
        if q_key in results:
            continue

        question = q_info.get("question", "")
        ground_truth = q_info.get("ground_truth", "")
        document_id = q_info.get("document_id", "")

        try:
            retrieved_docs = search_docs(question, embed_model, es_client)
            context_text = "\n\n".join(retrieved_docs)

            prompt = OPEN_PROMPT.format(
                context=context_text,
                question=question
            )

            answer_text = generate_answer_llama3(
                llama3_model,
                llama3_tokenizer,
                prompt
            )

            results[q_key] = {
                "question": question,
                "ground_truth": ground_truth,
                "document_id": document_id,
                "generated_answer": answer_text
            }

            if len(results) % CHECKPOINT_EVERY == 0:
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error on {q_key}: {e}")
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            continue

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved results:", len(results))

if __name__ == "__main__":
    main()
