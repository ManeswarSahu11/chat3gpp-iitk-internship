
import sys
from datetime import datetime
from elasticsearch import Elasticsearch
from retrievers.VectorRetriever import VectorRetrieval
from server.embedding import load_embeddings
from configs.model_configs import EMBED_CONFIG, SPLITTER_CONFIG
from langchain_text_splitters import RecursiveCharacterTextSplitter

VectorRetrieval.__abstractmethods__ = frozenset()

INDEX_NAME = "index_clean_rel1718_final_v2"

def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=SPLITTER_CONFIG["chunk_overlap"],
        length_function=len,
    )
    content = splitter.split_text(text)

    return {
        "text": content,
        "filename": file_path,
        "date": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python index_one_clean_rel1718_final_v2_file.py <file_path>")
        raise SystemExit(2)

    file_path = sys.argv[1]

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

    kb = VectorRetrieval(embed_model=embed_model, es_client=es_client)

    chunks = process_txt(file_path)
    if chunks["text"]:
        kb.build_index(INDEX_NAME, chunks)
        print(f"OK: {file_path}")
    else:
        print(f"EMPTY: {file_path}")

if __name__ == "__main__":
    main()
