
import re
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from configs.model_configs import MODEL_PATH, RERANKER_CONFIG

_RERANKER = None

def load_reranker(model_name=None, device=None):
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER

    model_name = model_name or RERANKER_CONFIG["reranker_model"]
    device = device or RERANKER_CONFIG["reranker_device"]

    model_path = MODEL_PATH["reranker_model"][model_name]
    _RERANKER = BGEM3FlagModel(
        model_path,
        use_fp16=True,
        use_multiprocessing=False,
        devices=device
    )
    return _RERANKER

def _clean_text(text: str, max_chars: int = 3000) -> str:
    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text[:max_chars].strip()

def _safe_colbert_vec(model, text: str):
    text = _clean_text(text)
    try:
        out = model.encode(
            [text],
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True
        )
        return np.asarray(out["colbert_vecs"][0])
    except BaseException:
        text = _clean_text(text, max_chars=1000)
        out = model.encode(
            [text],
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True
        )
        return np.asarray(out["colbert_vecs"][0])

def rerank_bgem3(query, docs, top_k=5):
    model = load_reranker()

    q_vec = _safe_colbert_vec(model, query)

    scored = []
    for doc in docs:
        try:
            d_vec = _safe_colbert_vec(model, doc)
            score = float(model.colbert_score(q_vec, d_vec))
        except BaseException:
            score = -1e9
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [x[0] for x in scored[:top_k]]
    top_scores = [x[1] for x in scored[:top_k]]
    return top_docs, top_scores
