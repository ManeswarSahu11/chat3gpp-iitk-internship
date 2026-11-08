
import sys
import re
from datetime import datetime
from elasticsearch import Elasticsearch
from retrievers.VectorRetriever import VectorRetrieval
from server.embedding import load_embeddings
from configs.model_configs import EMBED_CONFIG
from langchain_text_splitters import RecursiveCharacterTextSplitter

VectorRetrieval.__abstractmethods__ = frozenset()

INDEX_NAME = "index_clean"

def brutal_clean(text: str) -> str:
    text = str(text)
    text = text.replace("\x00", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]+', ' ', text)
    text = re.sub(r'\\begin\{[^}]+\}', ' ', text)
    text = re.sub(r'\\end\{[^}]+\}', ' ', text)
    text = re.sub(r'\\item\b', ' ', text)
    text = re.sub(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]+', ' ', text)   # ASCII-only fallback
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def main():
    if len(sys.argv) != 2:
        print("Usage: python salvage_one_clean_file.py <file_path>")
        raise SystemExit(2)

    file_path = sys.argv[1]

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text = brutal_clean(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=0,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    if not chunks:
        chunks = [text[:800]]

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

    payload = {
        "text": chunks,
        "filename": file_path,
        "date": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    }

    kb.build_index(INDEX_NAME, payload)
    print(f"SALVAGED: {file_path}")

if __name__ == "__main__":
    main()
