
import glob
import subprocess
import sys
from elasticsearch import Elasticsearch

INDEX_NAME = "index_clean_rel1718_final_v2"

def get_indexed_files(es):
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
        return set(b["key"] for b in res["aggregations"]["files"]["buckets"])
    except Exception:
        return set()

def main():
    es = Elasticsearch("http://localhost:9200", request_timeout=120, max_retries=3, retry_on_timeout=True)

    files = sorted(glob.glob("knowledge_base_cleaned_rel1718_final_v2/*.txt"))
    indexed_files = get_indexed_files(es)

    print("Found cleaned Rel-17/18 final_v2 files:", len(files), flush=True)
    print("Already indexed Rel-17/18 final_v2 files:", len(indexed_files), flush=True)

    done = 0
    failed = 0

    for i, path in enumerate(files, 1):
        if path in indexed_files:
            continue

        print(f"[{i}/{len(files)}] INDEX {path}", flush=True)

        try:
            r = subprocess.run(
                [sys.executable, "index_one_clean_rel1718_final_v2_file.py", path],
                timeout=1200
            )
            if r.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"FAILED: {path} (exit={r.returncode})", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAILED: {path} ({e})", flush=True)

    final_indexed = get_indexed_files(es)
    print("Newly indexed files:", done, flush=True)
    print("Failed files:", failed, flush=True)
    print("Final indexed Rel-17/18 final_v2 files:", len(final_indexed), flush=True)

if __name__ == "__main__":
    main()
