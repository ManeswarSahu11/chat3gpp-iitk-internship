# Project map

[Back to README](../README.md)

## Core components

| Path | Responsibility |
|---|---|
| `configs/model_configs.py` | Model paths, devices, and chunk settings |
| `configs/kb_configs.py` | Knowledge-base configuration |
| `retrievers/VectorRetriever.py` | Indexing, BM25, dense retrieval, RRF, and ColBERT reranking |
| `retrievers/base.py` | Retrieval interface |
| `server/embedding.py` | BGE-M3 embeddings |
| `server/llm.py` | Model loading and answer generation |
| `server/reranker.py` | BGE-M3 reranking helper |
| `utils.py` | Device and knowledge-base utilities |

## Preprocessing and indexing

| Script | Purpose |
|---|---|
| `build_clean_corpus.py` | Hierarchical text cleaning |
| `normalize_rel1718_final_v2.py` | Final-v2 normalization experiment |
| `init_database_clean_rel1718.py` | Release 17/18 index construction |
| `index_one_clean_rel1718_final_v2_file.py` | Single-file indexing |
| `resume_clean_index_rel1718_final_v2_exact.py` | Resume-based indexing |
| `resume_clean_index_subprocess.py` | Subprocess indexing workflow |
| `salvage_one_clean_file.py` | Fallback text cleanup and indexing |

## Evaluation scripts

| Experiment | Script |
|---|---|
| Llama-3 + RAG, TeleQnA | `run_teleqna_eval_clean_rel1718_final_v2.py` |
| Tele-it + RAG, TeleQnA | `run_tele_it_teleqna_clean_rel1718_final_v2.py` |
| Llama-3 + RAG, Tele-Eval | `run_teleeval_eval_clean_rel1718_final_v2.py` |
| Tele-it + RAG, Tele-Eval | `run_tele_it_teleeval_clean_rel1718_final_v2.py` |
| Tele-it no RAG, MCQ | `run_tele_it_teleqna_no_rag_gpu0.py` |
| Tele-it no RAG, paper-faithful prompt | `run_tele_it_teleqna_no_rag_paperfaithful.py` |
| Tele-it no RAG, open-ended | `run_teleit_teleeval_no_rag.py` |
| Mixtral judging | `run_mixtral_teleeval_judge_paperfaithful_v3.py` |

GPU-specific runners and alternative prompt scripts preserve the experiment variants.

## Data and experiment records

| Path | Contents |
|---|---|
| `knowledge_base_cleaned_rel1718_final/` | Cleaned Release 17/18 standards corpus |
| `data/teleqna/teleqna_paper.json` | TeleQnA MCQ subset |
| `data/teleeval_58627_exact.json` | Exact Tele-Eval subset |
| `release_map.json` | Release metadata |
| `teleqna_final_scores.txt` | TeleQnA score summary |
| `teleqna_completed_runs.txt` | Completed-run notes |
| `experiment_status_now.txt` | Experiment status notes |
| `preprocessing_comparison.txt` | Preprocessing comparison |
| `docs/Internship_Report_IITK.pdf` | Internship technical report |
