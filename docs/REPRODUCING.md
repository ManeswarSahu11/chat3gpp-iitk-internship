# Reproduction guide

[Back to README](../README.md)

This guide follows the code present in this snapshot. It requires a GPU research environment and manual configuration. Full historical result reproduction additionally requires the original frozen settings and completed artifacts described in the report.

## 1. Prepare the environment

Use Linux with an NVIDIA CUDA GPU and sufficient GPU memory for FP16 model inference plus BGE-M3. The included code does not establish a tested minimum memory requirement. Mixtral GGUF judging has separate memory and GPU-offload needs.

The historical dependency file contains Linux CUDA packages and is not a portable Windows installation recipe. Python 3.10 is a reasonable starting environment, but a clean installation has not been validated for this documentation change.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Imports in the source require packages missing from `requirements.txt`: **FlagEmbedding**, **accelerate**, and, for the GGUF judge, **llama-cpp-python**. Install compatible versions in the research environment and save the resolved versions with your run artifacts. A CUDA-enabled llama-cpp-python build is needed to use GPU offload; select its build for your hardware. Unpinned latest packages may conflict with the historical Torch and Transformers versions.

```bash
python -m pip install FlagEmbedding accelerate
# Optional: install a hardware-appropriate llama-cpp-python build for judging.
```

These are setup steps, not a tested dependency lock. If the historical pins fail on your hardware, resolve them in a separate environment before running long experiments.

## 2. Set model paths and devices

Edit [`configs/model_configs.py`](../configs/model_configs.py). Replace the original `/home/ugp/Downloads/...` paths with local model directories containing tokenizer and model files.

| Configuration key | Checkpoint |
|---|---|
| `MODEL_PATH["embed_model"]["bge-m3"]` | BAAI BGE-M3 |
| `MODEL_PATH["reranker_model"]["bge-m3"]` | BAAI BGE-M3 |
| `MODEL_PATH["llm_model"]["llama3-8B-instruct"]` | Meta Llama-3-8B-Instruct |
| `MODEL_PATH["llm_model"]["LLama-3-8B-Tele-it"]` | AliMaatouk Llama-3-8B-Tele-it |

Preserve the exact dictionary keys: capitalization is significant. Obtain checkpoints separately under their upstream access and license conditions.

The embedding loader in `server/embedding.py` hardcodes `cuda:0`; changing `EMBED_CONFIG` alone does not move BGE-M3. Several runners explicitly choose a GPU. The no-RAG Tele-Eval runner has its own absolute `MODEL_PATH` in `run_teleit_teleeval_no_rag.py`; update it separately. The v3 judge accepts a local Mixtral GGUF file through `--model` rather than the Transformers model configuration.

## 3. Prepare Elasticsearch

Provide a compatible Elasticsearch 8.x service; the Python client is pinned to 8.17.1. The scripts expect `http://localhost:9200` without authentication or TLS, so use an isolated local research service configured accordingly, or adapt all client constructors to your secured service.

```bash
curl http://localhost:9200
```

The server itself is not installed by `requirements.txt`. There is no Docker Compose file or Elasticsearch snapshot in this folder.

## 4. Align the corpus and index

**The supplied directory is `knowledge_base_cleaned_rel1718_final/`. The latest indexing scripts look for `knowledge_base_cleaned_rel1718_final_v2/`, which is absent.** Running them unchanged can find zero files.

For a fresh run using the supplied frozen corpus, make these small configuration edits before executing anything:

| File | Required edit |
|---|---|
| `resume_clean_index_rel1718_final_v2_exact.py` | Set `INDEX_NAME` to a fresh name such as `index_iitk_frozen_rebuild`; change its corpus glob to `knowledge_base_cleaned_rel1718_final/*.txt` |
| `index_one_clean_rel1718_final_v2_file.py` | Set the same `INDEX_NAME` |
| Each RAG evaluation runner you will use | Set the same `INDEX_NAME`; choose a new `RESULTS_FILE` for this run |

The filenames may still contain `v2`; the explicit corpus and index settings determine what gets evaluated. A fresh index name prevents mixing this run with an earlier corpus. This reconstructs an index from the supplied text, but does not guarantee the report's historical score.

To study the original v2 experiment instead, restore its original inputs and use the v2 settings. Do not relabel the frozen corpus as v2. `normalize_rel1718_final_v2.py` expects the missing `knowledge_base_cleaned_rel1718/` and deletes its destination directory at startup; it is not needed to index the supplied corpus.

Check the source count without loading models:

```bash
python -c "from pathlib import Path; print(len(list(Path('knowledge_base_cleaned_rel1718_final').glob('*.txt'))))"
# Expected: 1843
```

Build the aligned index:

```bash
python resume_clean_index_rel1718_final_v2_exact.py
```

The coordinator starts one subprocess per unindexed file with a 1,200-second timeout. Each worker loads BGE-M3, chunks a file, and adds successful embeddings. This trades startup cost for process isolation.

### Verify indexing before evaluation

```bash
curl 'http://localhost:9200/index_iitk_frozen_rebuild/_count'
curl -H 'Content-Type: application/json' \
  'http://localhost:9200/index_iitk_frozen_rebuild/_search' \
  -d '{"size":0,"aggs":{"files":{"terms":{"field":"filename","size":10000}}}}'
```

Inspect the file buckets and chunk counts. The report lists 254,570 chunks for its original frozen index; this is a historical reference, not a verified count for a new rebuild.

Resume checks operate at filename level. A partially indexed file can be considered complete, and embedding failures can silently omit chunks. Retrying the worker directly appends documents with generated IDs, so it can create duplicates. Audit completeness and use a fresh index for a clean rebuild rather than assuming the coordinator guarantees exact recovery.

## 5. Run TeleQnA

After index and output alignment:

```bash
# General Llama-3 with retrieved context
python run_teleqna_eval_clean_rel1718_final_v2.py

# Tele-it with retrieved context
python run_tele_it_teleqna_clean_rel1718_final_v2.py

# Tele-it without retrieval, paper-faithful prompt variant
python run_tele_it_teleqna_no_rag_paperfaithful.py
```

The input is `data/teleqna/teleqna_paper.json`, containing 1,514 questions. RAG retrieval uses question text only; answer options enter the generation prompt afterward.

The RAG scripts parse `option N` from both prediction and gold. Before trusting a score, verify that gold and prediction parsing succeeded. In the supplied code, `None == None` evaluates as correct if both fail; a robust scoring audit must reject missing parses. The report's corrected historical output is not included here.

## 6. Run Tele-Eval

```bash
python run_teleeval_eval_clean_rel1718_final_v2.py
python run_tele_it_teleeval_clean_rel1718_final_v2.py

# Update this script's independent MODEL_PATH first:
python run_teleit_teleeval_no_rag.py
```

Use `data/teleeval_58627_exact.json`. Expected complete count: **58,627**. The candidate and 58,629-entry files are historical alternatives, not interchangeable inputs for the report's exact evaluation.

The default v2 RAG output paths are:

```text
outputs/teleeval/chat3gpp_teleeval_results_clean_rel1718_final_v2.json
outputs/teleeval/tele_it_teleeval_results_clean_rel1718_final_v2.json
```

If you followed the fresh-run recommendation, substitute your new output names in all following commands.

## 7. Judge completed open-ended answers

Create the parent output directory first and provide your local quantized Mixtral GGUF path:

```bash
mkdir -p outputs/teleeval
python run_mixtral_teleeval_judge_paperfaithful_v3.py \
  --input outputs/teleeval/tele_it_teleeval_results_clean_rel1718_final_v2.json \
  --output outputs/teleeval/tele_it_rag_mixtral_judgments.json \
  --model /absolute/path/to/mixtral-instruct-q4.gguf \
  --n-gpu-layers 30 \
  --n-ctx 1024 \
  --n-threads 8 \
  --checkpoint-every 100
```

The judge compares the question, generated answer, and reference answer; a grammar constrains its answer to Yes or No. Empty, invalid, and failed judgments receive zero. Error records are checkpointed and skipped on resume, so review them before interpreting the final score. Judge context overflow and model quantization can affect acceptance.

## 8. Audit and preserve a run

For each run, preserve the corpus version, index name and counts, package versions, model checkpoint revisions, GPU details, prompt, generation settings, output count, parsing failures, and judge settings. Keep model or prompt changes in separate output files.

The main RAG runners resume by question ID. Their reported accuracy denominator is the number of saved results, so an incomplete run can print a seemingly final score. Compare saved IDs with all input IDs, not only the displayed percentage. A complete TeleQnA run has 1,514 results; a complete exact Tele-Eval run has 58,627 results.

## Common problems

| Symptom | Cause / action |
|---|---|
| Zero source files | Resolve the `final` versus `final_v2` directory mismatch |
| `ModuleNotFoundError: FlagEmbedding` | Add the missing dependency in the GPU environment |
| `device_map` loading error | Check accelerate installation and GPU configuration |
| Missing model directory | Replace absolute paths in config and standalone scripts |
| `KeyError: Llama-3-8B-Instruct` | Older no-RAG script uses a key absent from config; use `llama3-8B-instruct` |
| Cannot instantiate abstract retrieval class | Main runners clear `__abstractmethods__`; custom runners need the same workaround or an implemented `search()` method |
| Missing `index_one_clean_file.py` | Legacy resume helper references an absent worker; use the included exact v2 coordinator after alignment |
| Out-of-memory or slow search | Shared GPU loading and exhaustive dense scoring can be expensive; measure before scaling |
| Score differs from the report | Check corpus, model, prompt, parsing, completeness, index contents, and judge settings |

The documentation review did not run full GPU indexing or benchmark inference. It verified file roles, dataset counts, and the setup instructions against source.
