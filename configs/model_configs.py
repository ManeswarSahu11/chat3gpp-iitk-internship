EMBED_CONFIG = {
    "embed_model": "bge-m3",
    "embed_device": "cuda:0",
}

SPLITTER_CONFIG = {
    "chunk_size": 1250,
    "chunk_overlap": 0,
}

LLM_CONFIG = {
    "llm_model": "llama3-8B-instruct",
    "llm_device": "cuda:0",
    "dtype": "fp16",
}

RERANKER_CONFIG = {
    "reranker_model": "bge-m3",
    "reranker_device": "cuda:0",
}

MODEL_PATH = {
    "embed_model": {
        "bge-m3": "/home/ugp/Downloads/BAAI-bge-m3",
    },
    "llm_model": {
        "llama3-8B-instruct": "/home/ugp/Downloads/meta-llama-Meta-Llama-3-8B-Instruct",
        "LLama-3-8B-Tele-it": "/home/ugp/Downloads/AliMaatouk-LLama-3-8B-Tele-it",
        "Mixtral-8x7B-Instruct-v0.1": "/home/ugp/Downloads/mistralai-Mixtral-8x7B-Instruct-v0.1",
    },
    "reranker_model": {
        "bge-m3": "/home/ugp/Downloads/BAAI-bge-m3",
    }
}
