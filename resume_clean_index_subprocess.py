
import glob
import subprocess
from elasticsearch import Elasticsearch

INDEX_NAME = "index_clean"

def main():
    es = Elasticsearch("http://localhost:9200", request_timeout=120, max_retries=3, retry_on_timeout=True)

    try:
        res = es.search(
            index=INDEX_NAME,
            body={
                "size": 0,
                "aggs": {
                    "files": {
                        "terms": {
                            "field": "filename",
                            "size": 10000
                        }
                    }
                }
            }
        )
        indexed_files = set(b["key"] for b in res["aggregations"]["files"]["buckets"])
    except Exception:
        indexed_files = set()

    files = sorted(glob.glob("knowledge_base_cleaned/*.txt"))
    print("Found", len(files), "cleaned files")
    print("Already indexed cleaned files:", len(indexed_files))

    done = 0
    failed = 0

    for i, path in enumerate(files, 1):
        if path in indexed_files:
            continue

        print(f"[{i}/{len(files)}] Indexing {path}", flush=True)

        cmd = ["python", "index_one_clean_file.py", path]
        result = subprocess.run(cmd)

        if result.returncode == 0:
            done += 1
        else:
            failed += 1
            print(f"FAILED: {path} (exit={result.returncode})", flush=True)

    print("Newly indexed files:", done)
    print("Failed files:", failed)

if __name__ == "__main__":
    main()
