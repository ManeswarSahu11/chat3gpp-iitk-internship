# Execution guide

[Back to README](../README.md)

The internship report presents the following execution workflow in sections 25–37.

## Environment setup

Run the pipeline on the GPU server and activate the Python environment:

```bash
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install FlagEmbedding
pip install llama-cpp-python
```

## Model configuration

Configure local checkpoint paths in `configs/model_configs.py` and the embedding setup in `server/embedding.py`.

| Component | Model |
|---|---|
| Embedding | BAAI BGE-M3 |
| Reranking | BGE-M3 ColBERT-style late interaction |
| General generator | Llama-3-8B-Instruct |
| Telecom-adapted generator | Llama-3-8B-Tele-it |
| Open-ended evaluator | Quantized Mixtral-8x7B-Instruct GGUF |

## Corpus and index

| Setting | Final configuration in the report |
|---|---|
| Corpus | `knowledge_base_cleaned_rel1718_final/` |
| Source files | 1,843 |
| Elasticsearch index | `index_clean_rel1718_final` |
| Indexed chunks | 254,570 |
| Chunk size | 1,250 characters |
| Chunk overlap | 0 |

Check Elasticsearch and the final index:

```bash
curl http://localhost:9200
curl "http://localhost:9200/index_clean_rel1718_final/_count?pretty"
```

The report uses a resume-based index build, followed by source-file and chunk-count checks. The final and final-v2 corpora are separate experimental configurations.

## TeleQnA evaluation

Use `data/teleqna/teleqna_paper.json`, containing 1,514 questions.

1. Retrieve passages using the question text.
2. Add the question and answer options to the generation prompt.
3. Generate an answer with Llama-3-Instruct or Tele-it.
4. Extract the predicted option number and compare it with the reference option.

The experiments include Llama-3-Instruct with RAG, Tele-it with RAG, Tele-it without RAG, and a paper-faithful no-RAG prompt variant.

## Tele-Eval generation and judging

Use `data/teleeval_58627_exact.json`, containing 58,627 questions: 26,225 from Release 17 and 32,402 from Release 18.

Generate open-ended answers, then evaluate them using the local quantized Mixtral judge. The judge receives the question, reference answer, and generated answer, and returns Yes or No.

| Judge parameter | Setting |
|---|---:|
| Context window | 1,024 tokens |
| GPU layers | 30 |
| CPU threads | 8 |
| Checkpoint interval | 100 examples |

## Resume and monitoring

Generation scripts load existing result files and skip completed question IDs. Use a separate output filename for each experiment and preserve partial results to resume interrupted runs.

Expected completed counts are **1,514** for TeleQnA and **58,627** for Tele-Eval. The report describes monitoring logs, checking JSON result counts, and computing accuracy from completed result files.

See the [full report](Internship_Report_IITK.pdf) for the original server commands and the [project map](PROJECT_MAP.md) for the script names in this project.
