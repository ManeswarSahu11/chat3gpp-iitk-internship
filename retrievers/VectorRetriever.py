
from retrievers.base import BaseRetrieval
from elasticsearch.helpers import bulk
import numpy as np
import re

def _safe_clean_text(text, max_chars=3000):
    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r'\\begin\{[^}]+\}', ' ', text)
    text = re.sub(r'\\end\{[^}]+\}', ' ', text)
    text = re.sub(r'\\item\b', ' ', text)
    text = re.sub(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]

class VectorRetrieval(BaseRetrieval):
    def __init__(self, embed_model, es_client=None):
        super().__init__(embed_model=embed_model, es_client=es_client)
        self.embed_model = embed_model
        self.es_client = es_client

    def create_index(self, index_name):
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "default"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "filename": {"type": "keyword"},
                    "date": {"type": "date"}
                }
            }
        }
        if not self.es_client.indices.exists(index=index_name):
            self.es_client.indices.create(index=index_name, body=mapping)

    def build_index(self, index_name, chunks):
        self.create_index(index_name)

        texts = [_safe_clean_text(t) for t in chunks["text"]]
        filename = chunks["filename"]
        date = chunks["date"]

        good_texts = []
        good_embeddings = []

        try:
            output = self.embed_model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            embeddings = output["dense_vecs"]
            for text, emb in zip(texts, embeddings):
                good_texts.append(text)
                good_embeddings.append(emb)
        except BaseException:
            for text in texts:
                try:
                    output = self.embed_model.encode(
                        [text],
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False
                    )
                    emb = output["dense_vecs"][0]
                    good_texts.append(text)
                    good_embeddings.append(emb)
                except BaseException:
                    continue

        actions = []
        for text, emb in zip(good_texts, good_embeddings):
            actions.append({
                "_index": index_name,
                "_source": {
                    "content": text,
                    "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb),
                    "filename": filename,
                    "date": date
                }
            })

        if actions:
            bulk(self.es_client, actions)

    def bm25_retrieve(self, query, index_name, top_k):
        body = {
            "size": top_k,
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        return self.es_client.search(index=index_name, body=body)

    def vector_retrieve(self, query, index_name, top_k):
        q = self.embed_model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )["dense_vecs"][0]

        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": q.tolist() if hasattr(q, "tolist") else list(q)
                        }
                    }
                }
            }
        }
        return self.es_client.search(index=index_name, body=body)

    def calculate_rrf(self, bm25_ranks, vector_ranks, k=10):
        scores = {}

        for doc_id, rank in bm25_ranks:
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

        for doc_id, rank in vector_ranks:
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def get_documents_with_ids(self, ids, index_name):
        docs = []
        for doc_id in ids:
            res = self.es_client.get(index=index_name, id=doc_id)
            docs.append(res["_source"]["content"])
        return docs

    def rerank_colbert(self, query, docs):
        q = self.embed_model.encode(
            [_safe_clean_text(query, max_chars=1000)],
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True
        )["colbert_vecs"][0]

        scores = []
        for doc in docs:
            try:
                clean_doc = _safe_clean_text(doc, max_chars=1000)
                d = self.embed_model.encode(
                    [clean_doc],
                    return_dense=False,
                    return_sparse=False,
                    return_colbert_vecs=True
                )["colbert_vecs"]
                score = float(self.embed_model.colbert_score(q, d)[0])
            except BaseException:
                score = -1e9
            scores.append(score)

        return scores

    def search_rrf(self, index_name, query, top_k1=1000, top_k2=5, **kwargs):
        bm25_response = self.bm25_retrieve(query, index_name, top_k1)
        bm25_ranks = [(hit["_id"], idx + 1) for idx, hit in enumerate(bm25_response["hits"]["hits"])]

        vector_response = self.vector_retrieve(query, index_name, top_k1)
        vector_ranks = [(hit["_id"], idx + 1) for idx, hit in enumerate(vector_response["hits"]["hits"])]

        final_sorted_docs = self.calculate_rrf(bm25_ranks, vector_ranks, k=10)
        candidate_doc_ids = [doc_id for doc_id, _ in final_sorted_docs][:int(1/10 * top_k1)]

        docs = self.get_documents_with_ids(candidate_doc_ids, index_name)

        try:
            scores = self.rerank_colbert(query, docs)
        except BaseException:
            safe_docs = [_safe_clean_text(x, max_chars=1000) for x in docs]
            scores = self.rerank_colbert(query, safe_docs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        sorted_docs = [x[0] for x in ranked[:top_k2]]
        sorted_scores = [x[1] for x in ranked[:top_k2]]
        return sorted_docs, sorted_scores
