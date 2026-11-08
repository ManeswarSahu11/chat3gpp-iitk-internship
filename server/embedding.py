from configs.model_configs import MODEL_PATH, EMBED_CONFIG, SPLITTER_CONFIG
from utils import detect_device
from typing import Literal, List, Dict
from langchain.embeddings.base import Embeddings
import numpy as np

def embedding_device(device=None):
    device = device or EMBED_CONFIG["embed_device"]
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device

def load_embeddings(model, device):
    if model == "bge-m3":
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel(MODEL_PATH["embed_model"][model],
                              use_fp16=True,
                              device="cuda:0")
    else:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=MODEL_PATH["embed_model"][model],
                                     model_kwargs={"device": device})

def embed_documents(docs, embed_model):
    good_docs = []
    good_embeddings = []
    for doc in docs:
        try:
            out = embed_model.encode([doc], return_dense=True, return_sparse=False, return_colbert_vecs=False)
            good_docs.append(doc)
            good_embeddings.append(out["dense_vecs"][0])
        except Exception:
            pass
    if good_docs:
        return {"texts": good_docs, "embeddings": np.array(good_embeddings)}
    return {"texts": [], "embeddings": np.array([])}
