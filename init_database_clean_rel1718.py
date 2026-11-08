
import glob
from datetime import datetime
from elasticsearch import Elasticsearch
from retrievers.VectorRetriever import VectorRetrieval
from server.embedding import load_embeddings
from configs.model_configs import EMBED_CONFIG, SPLITTER_CONFIG
from langchain_text_splitters import RecursiveCharacterTextSplitter

VectorRetrieval.__abstractmethods__ = frozenset()

INDEX_NAME = "index_clean_rel1718"

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

    kb_files = sorted(glob.glob("knowledge_base_cleaned_rel1718/*.txt"))
    print(f"Found {len(kb_files)} Rel-17/18 cleaned files")

    for i, kb_file in enumerate(kb_files, 1):
        chunks = process_txt(kb_file)
        if chunks["text"]:
            print(f"[{i}/{len(kb_files)}] Adding {kb_file}")
            kb.build_index(INDEX_NAME, chunks)

if __name__ == "__main__":
    main()
