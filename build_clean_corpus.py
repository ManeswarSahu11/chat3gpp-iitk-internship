
import os
import re
import glob

SRC_DIR = "knowledge_base"
DST_DIR = "knowledge_base_cleaned"
os.makedirs(DST_DIR, exist_ok=True)

heading_re = re.compile(
    r'^\s*((?:\d+(?:\.\d+)*)|Annex\s+[A-Z](?:\.\d+)*)\s+(.+?)\s*$',
    re.IGNORECASE
)

def normalize_line(line: str) -> str:
    line = str(line)
    line = line.replace("\x00", " ")
    line = line.replace("\r", " ")
    line = line.replace("\t", " ")
    line = re.sub(r'\\begin\{[^}]+\}', ' ', line)
    line = re.sub(r'\\end\{[^}]+\}', ' ', line)
    line = re.sub(r'\\item\b', ' ', line)
    line = re.sub(r'\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?', ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()
    return line

def heading_level(tag: str) -> int:
    tag = tag.strip()
    if tag.lower().startswith("annex"):
        return 1
    return tag.count(".") + 1

def is_skip_heading(title: str, tag: str) -> bool:
    t = title.strip().lower()
    if tag.lower().startswith("annex"):
        return True
    if t in {"foreword", "contents", "references"}:
        return True
    if "table of contents" in t:
        return True
    return False

def process_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    lines = [normalize_line(x) for x in raw_lines]
    lines = [x for x in lines if x]

    sections = []
    stack = []
    current_text = []
    current_skip = True
    saw_real_heading = False

    def flush():
        nonlocal current_text, current_skip, stack, sections
        if stack and (not current_skip):
            content = " ".join(current_text).strip()
            if len(content) > 40:
                path_text = " > ".join(title for _, title, skip in stack if not skip)
                block = f"Title Path: {path_text}\nContent: {content}"
                sections.append(block)
        current_text = []

    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            tag = m.group(1).strip()
            title = m.group(2).strip()
            lvl = heading_level(tag)
            skip = is_skip_heading(title, tag)

            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title, skip))

            if not skip:
                saw_real_heading = True

            current_skip = (not saw_real_heading) or any(s for _, _, s in stack)
        else:
            if stack:
                current_text.append(line)

    flush()

    if not sections:
        text = " ".join(lines)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    return "\n\n".join(sections)

files = sorted(glob.glob(os.path.join(SRC_DIR, "*.txt")))
print("Input files:", len(files))

for i, path in enumerate(files, 1):
    cleaned = process_file(path)
    out_path = os.path.join(DST_DIR, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    if i % 200 == 0 or i == len(files):
        print(f"Wrote {i}/{len(files)}")

print("Done.")
