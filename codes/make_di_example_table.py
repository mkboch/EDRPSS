from pathlib import Path
import pandas as pd
import re
import html

# ============================================================
# Paths
# ============================================================

V3_ROOT = Path(".").resolve()

V2_DATASETS = Path(
    r"G:\My Drive\1. Studies\RPI\Thesis\1. Prof Ge Wang\3. Extra Papers\3. MoltBot\Paper\3. OpenClaw_PNAS\OpenClaw_V2\Datasets"
)

POSTS = V2_DATASETS / "posts.csv"

OUTDIR = V3_ROOT / "latex_tables"
OUTDIR.mkdir(exist_ok=True)

OUT_CSV = OUTDIR / "table_di_examples_source.csv"
OUT_TEX = OUTDIR / "table_di_examples.tex"

print("V3 root:", V3_ROOT)
print("Reading posts from:", POSTS)

if not POSTS.exists():
    raise FileNotFoundError(f"Missing posts.csv: {POSTS}")

# ============================================================
# DI-style lexical patterns
# This is for qualitative appendix calibration examples only.
# It selects posts with many directive/action-oriented markers.
# ============================================================

ACTION_PATTERNS = [
    r"\bmust\b",
    r"\bshould\b",
    r"\bneed to\b",
    r"\bhave to\b",
    r"\brequired to\b",
    r"\bmake sure\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bavoid\b",
    r"\bstop\b",
    r"\bplease\b",
    r"\buse\b",
    r"\brun\b",
    r"\bexecute\b",
    r"\binstall\b",
    r"\bdownload\b",
    r"\bopen\b",
    r"\bclick\b",
    r"\bcopy\b",
    r"\bpaste\b",
    r"\bdelete\b",
    r"\bremove\b",
    r"\bchange\b",
    r"\bupdate\b",
    r"\bcreate\b",
    r"\bwrite\b",
    r"\bsend\b",
    r"\bsubmit\b",
]

SENSITIVE_PATTERNS = [
    r"\bpassword\b",
    r"\btoken\b",
    r"\bsecret\b",
    r"\bcredential\b",
    r"\bapi key\b",
    r"\bprivate key\b",
    r"\badmin\b",
    r"\broot\b",
    r"\bsudo\b",
    r"\bpermission\b",
    r"\baccess\b",
    r"\bexecute\b",
    r"\bshell\b",
    r"\bterminal\b",
    r"\bscript\b",
    r"\bcommand\b",
]

ALL_PATTERNS = [(p, "action") for p in ACTION_PATTERNS] + [(p, "sensitive") for p in SENSITIVE_PATTERNS]

# ============================================================
# Helpers
# ============================================================

def clean_text(x):
    x = "" if pd.isna(x) else str(x)
    x = html.unescape(x)
    x = re.sub(r"https?://\S+|www\.\S+", "[URL]", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def latex_escape(s):
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in s)

def shorten_words(text, max_words=32):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def score_di(text):
    text_l = text.lower()
    action_matches = 0
    sensitive_matches = 0
    matched_terms = []

    for pat, fam in ALL_PATTERNS:
        if re.search(pat, text_l, flags=re.IGNORECASE):
            if fam == "action":
                action_matches += 1
            else:
                sensitive_matches += 1
            term = pat.replace(r"\b", "").replace("\\", "")
            matched_terms.append(term)

    raw = action_matches + sensitive_matches
    capped = min(raw, 10)
    return capped, action_matches, sensitive_matches, ", ".join(matched_terms[:8])

# ============================================================
# Read columns
# ============================================================

cols = list(pd.read_csv(POSTS, nrows=0).columns)
print("Detected columns:", cols)

id_col = find_col(cols, ["id", "post_id", "postId", "postID"])
title_col = find_col(cols, ["title", "post_title"])
content_col = find_col(cols, ["content", "body", "text", "post_text"])

if id_col is None:
    raise ValueError("Could not find post ID column.")
if title_col is None and content_col is None:
    raise ValueError("Could not find title/content/body/text column.")

usecols = [id_col] + [c for c in [title_col, content_col] if c]
print("Using columns:", usecols)

# ============================================================
# Stream posts and keep top examples
# ============================================================

top_chunks = []
chunk_size = 200000

for k, chunk in enumerate(pd.read_csv(POSTS, usecols=usecols, chunksize=chunk_size, low_memory=False), start=1):
    parts = []
    if title_col:
        parts.append(chunk[title_col].map(clean_text))
    if content_col:
        parts.append(chunk[content_col].map(clean_text))

    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p

    chunk["example_text"] = text.map(clean_text)
    chunk["word_count"] = chunk["example_text"].str.split().map(len)

    scores = chunk["example_text"].map(score_di)
    chunk["DI"] = scores.map(lambda x: x[0])
    chunk["action_matches"] = scores.map(lambda x: x[1])
    chunk["sensitive_matches"] = scores.map(lambda x: x[2])
    chunk["matched_terms"] = scores.map(lambda x: x[3])

    sub = chunk[(chunk["DI"] >= 4) & (chunk["word_count"] >= 20)].copy()

    if len(sub):
        top_chunks.append(
            sub[[id_col, "DI", "action_matches", "sensitive_matches", "matched_terms", "word_count", "example_text"]]
            .sort_values(["DI", "word_count"], ascending=[False, True])
            .head(100)
        )

    print(f"Processed chunk {k}, candidate rows so far: {sum(len(x) for x in top_chunks)}")

if not top_chunks:
    raise RuntimeError("No high-DI candidates found. Try lowering DI threshold from >=4 to >=3.")

cand = pd.concat(top_chunks, ignore_index=True)
cand = cand.drop_duplicates(subset=["example_text"])

# Avoid near-duplicate repeated templates
cand["prefix"] = cand["example_text"].str.lower().str[:160]
cand = cand.drop_duplicates(subset=["prefix"])

cand = cand.sort_values(["DI", "word_count"], ascending=[False, True]).head(12).copy()

cand["excerpt"] = cand["example_text"].map(lambda x: shorten_words(x, 32))

def interpret(row):
    if row["sensitive_matches"] > 0 and row["action_matches"] > 0:
        return "Mixed action and execution markers"
    if row["action_matches"] >= 4:
        return "Multiple action-oriented markers"
    if row["sensitive_matches"] >= 2:
        return "Execution/sensitive markers"
    return "Directive-marker-rich wording"

cand["interpretation"] = cand.apply(interpret, axis=1)

out = pd.DataFrame({
    "example": range(1, len(cand) + 1),
    "post_id": cand[id_col].astype(str).values,
    "DI": cand["DI"].astype(int).values,
    "words": cand["word_count"].astype(int).values,
    "matched_terms": cand["matched_terms"].values,
    "excerpt": cand["excerpt"].values,
    "interpretation": cand["interpretation"].values,
})

out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Illustrative high-DI post excerpts.} Examples are shortened excerpts from posts with high directive-marker burden. DI reflects matched directive-marker burden, not intent, harmfulness, or downstream execution.}
\label{tab:di_examples}
\begin{tabular}{@{}rrrrp{0.38\linewidth}p{0.24\linewidth}@{}}
\toprule
Ex. & Post ID & DI & Words & Short excerpt & Interpretation \\
\midrule
""")
    for _, r in out.iterrows():
        f.write(
            f"{int(r['example'])} & {latex_escape(r['post_id'])} & {int(r['DI'])} & {int(r['words'])} & "
            f"{latex_escape(r['excerpt'])} & {latex_escape(r['interpretation'])} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

print("\nSaved CSV:", OUT_CSV)
print("Saved LaTeX table:", OUT_TEX)
print("\nPreview:")
print(out[["example", "post_id", "DI", "words", "matched_terms", "excerpt"]].to_string(index=False))
