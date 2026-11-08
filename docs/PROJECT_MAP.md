# Project map

[Back to README](../README.md)

## Configuration and shared code

| Path | Responsibility |
|---|---|
| `configs/model_configs.py` | Local model directories, device defaults, 1,250-character chunking |
| `configs/kb_configs.py` | Legacy knowledge-base root, distinct from the final corpus directory |
| `retrievers/VectorRetriever.py` | Index mapping, embedding insertion, BM25, dense scoring, RRF, ColBERT reranking |
| `retrievers/base.py` | Abstract interface; `search()` remains unimplemented by the concrete retriever |
| `server/embedding.py` | BGE-M3 loading, with a hardcoded CUDA device |
| `server/llm.py` | Seed setup, FP16 model loading, greedy generation |
| `server/reranker.py` | Separate cached reranker helper; not called by the main `search_rrf` path |
| `utils.py` | Device detection and older folder utilities |

## Corpus preparation and indexing

| Script | Role / status |
|---|---|
| `build_clean_corpus.py` | Earlier heading-aware cleanup; expects missing raw `knowledge_base/` |
| `normalize_rel1718_final_v2.py` | Experimental normalization; expects a missing earlier corpus and replaces its destination |
| `index_one_clean_rel1718_final_v2_file.py` | Included one-file indexing worker; target index defaults to v2 |
| `resume_clean_index_rel1718_final_v2_exact.py` | Included coordinator; skips filenames already represented in the index |
| `init_database_clean_rel1718.py` | Older bulk file loop targeting `knowledge_base_cleaned_rel1718/` |
| `resume_clean_index_subprocess.py` | Legacy helper; references absent `index_one_clean_file.py` |
| `salvage_one_clean_file.py` | Older ASCII cleanup and 800-character fallback targeting `index_clean` |

See [the reproduction guide](REPRODUCING.md#4-align-the-corpus-and-index) before launching the worker or coordinator with the supplied final corpus.

## Main experiment entry points

| Experiment | Script |
|---|---|
| Llama-3 + RAG, TeleQnA | `run_teleqna_eval_clean_rel1718_final_v2.py` |
| Tele-it + RAG, TeleQnA | `run_tele_it_teleqna_clean_rel1718_final_v2.py` |
| Llama-3 + RAG, Tele-Eval | `run_teleeval_eval_clean_rel1718_final_v2.py` |
| Tele-it + RAG, Tele-Eval | `run_tele_it_teleeval_clean_rel1718_final_v2.py` |
| Tele-it no RAG, paper-faithful MCQ prompt | `run_tele_it_teleqna_no_rag_paperfaithful.py` |
| Tele-it no RAG, open-ended | `run_teleit_teleeval_no_rag.py` |
| Grammar-constrained GGUF judging | `run_mixtral_teleeval_judge_paperfaithful_v3.py` |

## Preserved variants

GPU-specific copies and alternate prompts are experimental history, not interchangeable launch commands:

- `run_tele_it_teleqna_clean_rel1718_final_v2_gpu0.py` and `run_tele_it_teleeval_clean_rel1718_final_gpu*.py`: device-specific variants; inspect their corpus/index/output constants.
- `run_tele_it_teleqna_no_rag_gpu0.py` and `run_tele_it_teleqna_pure.py`: alternate no-RAG MCQ prompts and parsers.
- `run_llama3_teleqna_no_rag_gpu0.py`: uses an LLM key absent from the supplied configuration and starts a fresh result dictionary.
- `run_teleeval_eval.py` and `run_mixtral_teleeval_judge_single_fixed.py`: earlier generation and judging variants.
- `teleit_sanity_check.py`: model sanity check using its own environment assumptions.

## Data and provenance artifacts

`knowledge_base_cleaned_rel1718_final/` holds 1,843 cleaned text files. `data/` holds benchmark inputs, and `release_map.json` holds release metadata. The source mappings and runner behavior do not establish per-clause citations or runtime release filtering.

`teleqna_final_scores.txt`, `teleqna_completed_runs.txt`, `experiment_status_now.txt`, `final_run_order.txt`, and `next_runs.txt` preserve experiment notes. `preprocessing_comparison.txt` preserves a larger preprocessing comparison. Read them alongside the final report because they describe different stages of the work.

`docs/Internship_Report_IITK.pdf` is an unchanged copy of the report supplied by the author. This repository does not include the other project folders, checkpoint weights, prediction archives, or Elasticsearch database described in that report.
