from pathlib import Path
import pandas as pd
import re

ROOT = Path(".").resolve()

A1 = ROOT / "human_validation_sample_400_ANNOTATED_annotator1.csv"
A2 = ROOT / "human_validation_sample_400_ANNOTATED_annotator2.csv"
OUTDIR = ROOT / "human_validation_checks"
OUTDIR.mkdir(exist_ok=True)

allowed_labels = {"corrective", "affirmation", "adversarial", "neutral", "unclear"}
allowed_binary = {"yes", "no", "unclear"}

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x

def load_file(path, annotator_num):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = [
        "annotation_id",
        "comment_text",
        f"human_label_annotator{annotator_num}",
        f"is_corrective_annotator{annotator_num}",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name} is missing columns: {missing_cols}")

    df["annotation_id"] = df["annotation_id"].astype(str).str.strip()
    df["comment_text_norm"] = df["comment_text"].map(norm)
    df[f"human_label_annotator{annotator_num}_norm"] = df[f"human_label_annotator{annotator_num}"].map(norm)
    df[f"is_corrective_annotator{annotator_num}_norm"] = df[f"is_corrective_annotator{annotator_num}"].map(norm)
    return df

df1 = load_file(A1, 1)
df2 = load_file(A2, 2)

lines = []
lines.append("=" * 90)
lines.append("HUMAN VALIDATION ANNOTATION CHECK")
lines.append("=" * 90)
lines.append(f"Root: {ROOT}")
lines.append(f"Annotator 1 file: {A1.name}")
lines.append(f"Annotator 2 file: {A2.name}")
lines.append("")

# Basic row checks
lines.append("BASIC ROW CHECKS")
lines.append("-" * 90)
lines.append(f"Annotator 1 rows: {len(df1)}")
lines.append(f"Annotator 2 rows: {len(df2)}")
lines.append(f"Annotator 1 unique annotation_id: {df1['annotation_id'].nunique()}")
lines.append(f"Annotator 2 unique annotation_id: {df2['annotation_id'].nunique()}")

dup1 = df1[df1["annotation_id"].duplicated(keep=False)]
dup2 = df2[df2["annotation_id"].duplicated(keep=False)]
lines.append(f"Annotator 1 duplicate annotation_id rows: {len(dup1)}")
lines.append(f"Annotator 2 duplicate annotation_id rows: {len(dup2)}")
lines.append("")

# Missing annotation checks
for num, df in [(1, df1), (2, df2)]:
    label_col = f"human_label_annotator{num}_norm"
    corr_col = f"is_corrective_annotator{num}_norm"

    missing_label = df[df[label_col].eq("")]
    missing_corr = df[df[corr_col].eq("")]
    bad_label = df[~df[label_col].isin(allowed_labels)]
    bad_corr = df[~df[corr_col].isin(allowed_binary)]

    lines.append(f"ANNOTATOR {num} FIELD CHECKS")
    lines.append("-" * 90)
    lines.append(f"Missing human_label_annotator{num}: {len(missing_label)}")
    lines.append(f"Missing is_corrective_annotator{num}: {len(missing_corr)}")
    lines.append(f"Invalid human_label_annotator{num}: {len(bad_label)}")
    lines.append(f"Invalid is_corrective_annotator{num}: {len(bad_corr)}")
    lines.append("")
    lines.append(f"human_label_annotator{num} distribution:")
    lines.append(str(df[label_col].value_counts(dropna=False)))
    lines.append("")
    lines.append(f"is_corrective_annotator{num} distribution:")
    lines.append(str(df[corr_col].value_counts(dropna=False)))
    lines.append("")

    if len(missing_label) or len(missing_corr) or len(bad_label) or len(bad_corr):
        problem = pd.concat([
            missing_label.assign(problem=f"missing human_label_annotator{num}"),
            missing_corr.assign(problem=f"missing is_corrective_annotator{num}"),
            bad_label.assign(problem=f"invalid human_label_annotator{num}"),
            bad_corr.assign(problem=f"invalid is_corrective_annotator{num}"),
        ], ignore_index=True).drop_duplicates(subset=["annotation_id", "problem"])
        problem.to_csv(OUTDIR / f"annotator{num}_field_problems.csv", index=False, encoding="utf-8-sig")

# Compare IDs and text
ids1 = set(df1["annotation_id"])
ids2 = set(df2["annotation_id"])
only1 = sorted(ids1 - ids2, key=lambda x: int(x) if x.isdigit() else x)
only2 = sorted(ids2 - ids1, key=lambda x: int(x) if x.isdigit() else x)

lines.append("CROSS-FILE MATCH CHECKS")
lines.append("-" * 90)
lines.append(f"IDs only in annotator 1 file: {len(only1)}")
lines.append(f"IDs only in annotator 2 file: {len(only2)}")
if only1:
    lines.append(f"First IDs only in annotator 1: {only1[:20]}")
if only2:
    lines.append(f"First IDs only in annotator 2: {only2[:20]}")
lines.append("")

merged = df1[[
    "annotation_id",
    "comment_text",
    "comment_text_norm",
    "human_label_annotator1_norm",
    "is_corrective_annotator1_norm"
]].merge(
    df2[[
        "annotation_id",
        "comment_text",
        "comment_text_norm",
        "human_label_annotator2_norm",
        "is_corrective_annotator2_norm"
    ]],
    on="annotation_id",
    how="inner",
    suffixes=("_a1", "_a2")
)

merged["comment_text_same"] = merged["comment_text_norm_a1"].eq(merged["comment_text_norm_a2"])
text_mismatch = merged[~merged["comment_text_same"]].copy()

lines.append(f"Matched annotation_id rows: {len(merged)}")
lines.append(f"Comment text mismatches across files: {len(text_mismatch)}")
lines.append("")

if len(text_mismatch):
    text_mismatch.to_csv(OUTDIR / "comment_text_mismatches.csv", index=False, encoding="utf-8-sig")

# Agreement / variation checks
merged["label_same"] = merged["human_label_annotator1_norm"].eq(merged["human_label_annotator2_norm"])
merged["is_corrective_same"] = merged["is_corrective_annotator1_norm"].eq(merged["is_corrective_annotator2_norm"])

label_disagree = merged[~merged["label_same"]].copy()
corr_disagree = merged[~merged["is_corrective_same"]].copy()

lines.append("ANNOTATOR VARIATION / AGREEMENT CHECKS")
lines.append("-" * 90)
lines.append(f"Same human_label: {merged['label_same'].sum()} / {len(merged)} ({merged['label_same'].mean():.3f})")
lines.append(f"Different human_label: {len(label_disagree)} / {len(merged)} ({len(label_disagree)/len(merged):.3f})")
lines.append(f"Same is_corrective: {merged['is_corrective_same'].sum()} / {len(merged)} ({merged['is_corrective_same'].mean():.3f})")
lines.append(f"Different is_corrective: {len(corr_disagree)} / {len(merged)} ({len(corr_disagree)/len(merged):.3f})")
lines.append("")

lines.append("Human-label crosstab, rows=annotator1, cols=annotator2:")
label_ct = pd.crosstab(
    merged["human_label_annotator1_norm"],
    merged["human_label_annotator2_norm"],
    dropna=False
)
lines.append(str(label_ct))
lines.append("")

lines.append("is_corrective crosstab, rows=annotator1, cols=annotator2:")
corr_ct = pd.crosstab(
    merged["is_corrective_annotator1_norm"],
    merged["is_corrective_annotator2_norm"],
    dropna=False
)
lines.append(str(corr_ct))
lines.append("")

label_disagree.to_csv(OUTDIR / "label_disagreements_between_annotators.csv", index=False, encoding="utf-8-sig")
corr_disagree.to_csv(OUTDIR / "is_corrective_disagreements_between_annotators.csv", index=False, encoding="utf-8-sig")
merged.to_csv(OUTDIR / "merged_annotation_check.csv", index=False, encoding="utf-8-sig")

# Save crosstabs
label_ct.to_csv(OUTDIR / "human_label_crosstab.csv", encoding="utf-8-sig")
corr_ct.to_csv(OUTDIR / "is_corrective_crosstab.csv", encoding="utf-8-sig")

report = "\n".join(lines)
print(report)

(OUTDIR / "annotation_check_report.txt").write_text(report, encoding="utf-8")

print("\nSaved check outputs in:")
print(OUTDIR)
print("\nKey files:")
print(OUTDIR / "annotation_check_report.txt")
print(OUTDIR / "label_disagreements_between_annotators.csv")
print(OUTDIR / "is_corrective_disagreements_between_annotators.csv")
