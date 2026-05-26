from pathlib import Path
import sys
import re
import pandas as pd

V3_ROOT = Path(r"G:\My Drive\1. Studies\RPI\Thesis\1. Prof Ge Wang\3. Extra Papers\3. MoltBot\Paper\OpenClaw_V3")
sys.path.insert(0, str(V3_ROOT / "codes"))

from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS

POSTS_IN = Path(r"F:\Research Files\OpenClaw_V3\moltbook_csv\posts_archive.csv")
OUTDIR = Path(r"F:\Research Files\OpenClaw_V3\working_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "posts_with_di.csv"

DI_CAP = 10
CHUNKSIZE = 100000

def compute_di_text(title, content):
    title = "" if pd.isna(title) else str(title)
    content = "" if pd.isna(content) else str(content)
    t = (title + " " + content).lower()
    a = sum(1 for pat in ACTION_PATTERNS if re.search(pat, t))
    s = sum(1 for pat in SENSITIVE_PATTERNS if re.search(pat, t))
    return min(a + s, DI_CAP)

print("=" * 100)
print("CREATE F-DRIVE posts_with_di.csv")
print("=" * 100)
print("INPUT:", POSTS_IN)
print("OUTPUT:", OUT)

if not POSTS_IN.exists():
    raise FileNotFoundError(POSTS_IN)

first = True
total = 0
di_counts = {}

for k, chunk in enumerate(pd.read_csv(POSTS_IN, chunksize=CHUNKSIZE, low_memory=False), start=1):
    if "title" not in chunk.columns:
        chunk["title"] = ""
    if "content" not in chunk.columns:
        chunk["content"] = ""

    chunk["di_post"] = [
        compute_di_text(t, c)
        for t, c in zip(chunk["title"], chunk["content"])
    ]

    for val, cnt in chunk["di_post"].value_counts().items():
        di_counts[int(val)] = di_counts.get(int(val), 0) + int(cnt)

    chunk.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    total += len(chunk)

    print(f"chunk {k:04d} | rows={len(chunk):,} | total={total:,}")

print("\nDONE")
print("Total rows written:", total)
print("DI counts:")
for k in sorted(di_counts):
    print(f"DI={k}: {di_counts[k]:,}")
print("Saved:", OUT)
