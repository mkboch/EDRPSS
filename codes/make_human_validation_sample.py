from pathlib import Path
import pandas as pd
import re

ROOT = Path(".").resolve()
INFILE = ROOT / "comments_labeled.csv"
OUTDIR = ROOT / "human_validation"
OUTDIR.mkdir(exist_ok=True)

OUTFILE_BLINDED = OUTDIR / "human_validation_sample_400_BLINDED.csv"
OUTFILE_KEY = OUTDIR / "human_validation_sample_400_KEY_DO_NOT_SHARE.csv"

print("Root:", ROOT)
print("Input:", INFILE)

if not INFILE.exists():
    raise FileNotFoundError(f"Missing input file: {INFILE}")

cols = list(pd.read_csv(INFILE, nrows=0).columns)
print("Columns:", cols)

if "response_type" not in cols:
    raise ValueError("Missing required column: response_type")

text_candidates = [
    "comment_text", "text", "content", "body", "comment", "message",
    "comment_body", "comment_content"
]
text_col = None
for c in text_candidates:
    if c in cols:
        text_col = c
        break

if text_col is None:
    raise ValueError(f"Could not find comment text column. Columns are: {cols}")

id_col = None
for c in ["comment_id", "id", "commentId", "commentID"]:
    if c in cols:
        id_col = c
        break

usecols = ["response_type", text_col]
if id_col:
    usecols.append(id_col)

df = pd.read_csv(INFILE, usecols=usecols, low_memory=False)

df["response_type_norm"] = df["response_type"].astype(str).str.strip().str.lower()
df[text_col] = df[text_col].astype(str)

def clean_text(x):
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

df["comment_text_for_annotation"] = df[text_col].map(clean_text)
df = df[df["comment_text_for_annotation"].str.len() >= 10].copy()

# Local file has only 86 adversarial rows, so we use all adversarial rows
# and distribute the remaining rows across other classes.
target_counts = {
    "corrective": 105,
    "affirmation": 105,
    "adversarial": 86,
    "neutral": 104,
}

samples = []
availability = {}

for label, n in target_counts.items():
    sub = df[df["response_type_norm"].eq(label)].copy()
    availability[label] = len(sub)
    print(label, "available:", len(sub), "target:", n)
    if len(sub) < n:
        raise ValueError(f"Not enough rows for {label}: need {n}, found {len(sub)}")
    samples.append(sub.sample(n=n, random_state=20260520))

sample = pd.concat(samples, ignore_index=True)
sample = sample.sample(frac=1, random_state=20260521).reset_index(drop=True)

out = pd.DataFrame()
out["annotation_id"] = range(1, len(sample) + 1)
out["comment_text"] = sample["comment_text_for_annotation"].values

out["human_label_annotator1"] = ""
out["is_corrective_annotator1"] = ""
out["human_label_annotator2"] = ""
out["is_corrective_annotator2"] = ""
out["notes"] = ""

out.to_csv(OUTFILE_BLINDED, index=False, encoding="utf-8-sig")

key = pd.DataFrame()
key["annotation_id"] = out["annotation_id"]
if id_col:
    key["comment_id"] = sample[id_col].values
else:
    key["comment_id"] = ""
key["machine_label"] = sample["response_type_norm"].values
key["comment_text"] = sample["comment_text_for_annotation"].values
key.to_csv(OUTFILE_KEY, index=False, encoding="utf-8-sig")

print("\nSaved blinded annotation file:")
print(OUTFILE_BLINDED)

print("\nSaved key file. DO NOT SHARE WITH ANNOTATORS:")
print(OUTFILE_KEY)

print("\nMachine-label distribution in hidden key:")
print(key["machine_label"].value_counts())

print("\nAvailability in source file:")
for k, v in availability.items():
    print(f"{k}: {v}")

print("\nDone.")
