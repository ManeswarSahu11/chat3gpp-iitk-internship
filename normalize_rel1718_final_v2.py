
import os
import re
import glob
import shutil

SRC_DIR = "knowledge_base_cleaned_rel1718"
DST_DIR = "knowledge_base_cleaned_rel1718_final_v2"

if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)
os.makedirs(DST_DIR, exist_ok=True)

def normalize_text(text: str) -> str:
    text = str(text)

    # conservative cleanup
    text = text.replace("\x00", " ")
    text = text.replace("~", " ")
    text = text.replace("\\_", "_")
    text = text.replace("{", " ")
    text = text.replace("}", " ")

    # remove common latex structural remnants
    text = re.sub(r'\\begin\{[^}]+\}', ' ', text)
    text = re.sub(r'\\end\{[^}]+\}', ' ', text)
    text = re.sub(r'\\item\b', ' ', text)
    text = re.sub(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?', ' ', text)

    # keep bracketed references like [4], [14], [A.2]
    # just normalize surrounding whitespace
    text = re.sub(r'\[\s*([0-9A-Za-z.\-]+)\s*\]', r'[\1]', text)

    # make broken API/path-ish escapes a bit more natural
    text = re.sub(r'\\\s*([A-Za-z])', r' \1', text)
    text = re.sub(r'\s*/\s*', ' / ', text)

    # normalize whitespace line-by-line to preserve title path structure
    lines = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines).strip()

files = sorted(glob.glob(os.path.join(SRC_DIR, "*.txt")))
print("Input files:", len(files))

for i, path in enumerate(files, 1):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    cleaned = normalize_text(text)

    out_path = os.path.join(DST_DIR, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    if i % 200 == 0 or i == len(files):
        print(f"Wrote {i}/{len(files)}")

print("Done.")
