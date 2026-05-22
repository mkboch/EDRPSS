from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(".").resolve()

A1 = ROOT / "human_validation_sample_400_ANNOTATED_annotator1.csv"
A2 = ROOT / "human_validation_sample_400_ANNOTATED_annotator2.csv"
KEY = ROOT / "human_validation_sample_400_KEY_DO_NOT_SHARE.csv"

OUTDIR = ROOT / "human_validation_results"
OUTDIR.mkdir(exist_ok=True)

allowed_labels = ["corrective", "affirmation", "adversarial", "neutral", "unclear"]

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x

def cohen_kappa(y1, y2, labels):
    y1 = list(y1)
    y2 = list(y2)
    n = len(y1)
    if n == 0:
        return np.nan

    obs = sum(a == b for a, b in zip(y1, y2)) / n

    p1 = {lab: y1.count(lab) / n for lab in labels}
    p2 = {lab: y2.count(lab) / n for lab in labels}
    exp = sum(p1[lab] * p2[lab] for lab in labels)

    if abs(1 - exp) < 1e-12:
        return np.nan
    return (obs - exp) / (1 - exp)

def binary_metrics(y_true, y_pred, positive="yes"):
    y_true = list(y_true)
    y_pred = list(y_pred)

    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    tn = sum(t != positive and p != positive for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "N": len(y_true),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

def multiclass_metrics(y_true, y_pred, labels):
    rows = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        support = sum(t == lab for t in y_true)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append({
            "label": lab,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)

# Load files
df1 = pd.read_csv(A1, dtype=str, keep_default_na=False)
df2 = pd.read_csv(A2, dtype=str, keep_default_na=False)
key = pd.read_csv(KEY, dtype=str, keep_default_na=False)

# Normalize
df1["annotation_id"] = df1["annotation_id"].astype(str).str.strip()
df2["annotation_id"] = df2["annotation_id"].astype(str).str.strip()
key["annotation_id"] = key["annotation_id"].astype(str).str.strip()

df1["human_label_annotator1_norm"] = df1["human_label_annotator1"].map(norm)
df1["is_corrective_annotator1_norm"] = df1["is_corrective_annotator1"].map(norm)

df2["human_label_annotator2_norm"] = df2["human_label_annotator2"].map(norm)
df2["is_corrective_annotator2_norm"] = df2["is_corrective_annotator2"].map(norm)

key["machine_label_norm"] = key["machine_label"].map(norm)
key["machine_is_corrective"] = key["machine_label_norm"].map(lambda x: "yes" if x == "corrective" else "no")

# Merge
merged = key.merge(
    df1[["annotation_id", "human_label_annotator1_norm", "is_corrective_annotator1_norm"]],
    on="annotation_id",
    how="inner"
).merge(
    df2[["annotation_id", "human_label_annotator2_norm", "is_corrective_annotator2_norm"]],
    on="annotation_id",
    how="inner"
)

# Inter-annotator agreement
label_kappa = cohen_kappa(
    merged["human_label_annotator1_norm"],
    merged["human_label_annotator2_norm"],
    ["corrective", "affirmation", "adversarial", "neutral", "unclear"]
)

binary_kappa = cohen_kappa(
    merged["is_corrective_annotator1_norm"],
    merged["is_corrective_annotator2_norm"],
    ["yes", "no", "unclear"]
)

label_agreement = (merged["human_label_annotator1_norm"] == merged["human_label_annotator2_norm"]).mean()
binary_agreement = (merged["is_corrective_annotator1_norm"] == merged["is_corrective_annotator2_norm"]).mean()

# Binary corrective metrics: machine corrective vs each annotator
m1 = binary_metrics(merged["is_corrective_annotator1_norm"], merged["machine_is_corrective"])
m2 = binary_metrics(merged["is_corrective_annotator2_norm"], merged["machine_is_corrective"])

# Agreement-only subset for binary corrective
agree_binary = merged[
    merged["is_corrective_annotator1_norm"].eq(merged["is_corrective_annotator2_norm"])
].copy()
magree = binary_metrics(agree_binary["is_corrective_annotator1_norm"], agree_binary["machine_is_corrective"])

binary_summary = pd.DataFrame([
    {"reference": "Annotator 1", **m1},
    {"reference": "Annotator 2", **m2},
    {"reference": "Agreement-only subset", **magree},
])

binary_summary.to_csv(OUTDIR / "binary_corrective_validation_metrics.csv", index=False)

# Multiclass metrics: machine label vs annotator 1 and annotator 2
labels4 = ["corrective", "affirmation", "adversarial", "neutral"]

mc1 = multiclass_metrics(
    merged["human_label_annotator1_norm"],
    merged["machine_label_norm"],
    labels4
)
mc1["reference"] = "Annotator 1"

mc2 = multiclass_metrics(
    merged["human_label_annotator2_norm"],
    merged["machine_label_norm"],
    labels4
)
mc2["reference"] = "Annotator 2"

multi_summary = pd.concat([mc1, mc2], ignore_index=True)
multi_summary = multi_summary[["reference", "label", "support", "precision", "recall", "f1"]]
multi_summary.to_csv(OUTDIR / "multiclass_label_validation_metrics.csv", index=False)

# Crosstabs
pd.crosstab(
    merged["human_label_annotator1_norm"],
    merged["human_label_annotator2_norm"]
).to_csv(OUTDIR / "interannotator_label_crosstab.csv")

pd.crosstab(
    merged["is_corrective_annotator1_norm"],
    merged["is_corrective_annotator2_norm"]
).to_csv(OUTDIR / "interannotator_is_corrective_crosstab.csv")

pd.crosstab(
    merged["is_corrective_annotator1_norm"],
    merged["machine_is_corrective"]
).to_csv(OUTDIR / "machine_vs_annotator1_binary_crosstab.csv")

pd.crosstab(
    merged["is_corrective_annotator2_norm"],
    merged["machine_is_corrective"]
).to_csv(OUTDIR / "machine_vs_annotator2_binary_crosstab.csv")

# LaTeX table for paper
table_path = OUTDIR / "table_human_validation.tex"
with open(table_path, "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Human validation of deterministic corrective labels.} A stratified sample of 400 comments was independently annotated by two human annotators. Metrics evaluate whether deterministic corrective labels align with human judgments of correction-like feedback.}
\label{tab:human_validation}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Reference & $N$ & Precision & Recall & F1 \\
\midrule
""")
    for _, row in binary_summary.iterrows():
        f.write(
            f"{row['reference']} & {int(row['N'])} & "
            f"{row['precision']:.3f} & {row['recall']:.3f} & {row['f1']:.3f} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}

\vspace{0.3em}
\footnotesize Inter-annotator agreement was """
            + f"{binary_agreement:.3f} for the binary corrective judgment "
            + f"(Cohen's $\\kappa={binary_kappa:.3f}$) and "
            + f"{label_agreement:.3f} for the five-way response label "
            + f"(Cohen's $\\kappa={label_kappa:.3f}$)."
            + "\n\\end{table}\n")

# Summary report
report_lines = []
report_lines.append("=" * 90)
report_lines.append("HUMAN VALIDATION METRICS")
report_lines.append("=" * 90)
report_lines.append(f"N merged rows: {len(merged)}")
report_lines.append("")
report_lines.append("INTER-ANNOTATOR AGREEMENT")
report_lines.append("-" * 90)
report_lines.append(f"Five-way label agreement: {label_agreement:.3f}")
report_lines.append(f"Five-way label Cohen kappa: {label_kappa:.3f}")
report_lines.append(f"Binary corrective agreement: {binary_agreement:.3f}")
report_lines.append(f"Binary corrective Cohen kappa: {binary_kappa:.3f}")
report_lines.append("")
report_lines.append("BINARY CORRECTIVE VALIDATION METRICS")
report_lines.append("-" * 90)
report_lines.append(binary_summary.to_string(index=False))
report_lines.append("")
report_lines.append("MULTICLASS VALIDATION METRICS")
report_lines.append("-" * 90)
report_lines.append(multi_summary.to_string(index=False))

report = "\n".join(report_lines)
print(report)

(OUTDIR / "human_validation_metrics_report.txt").write_text(report, encoding="utf-8")
merged.to_csv(OUTDIR / "merged_human_validation_with_machine_labels.csv", index=False, encoding="utf-8-sig")

print("\nSaved outputs in:")
print(OUTDIR)
print("\nMain report:")
print(OUTDIR / "human_validation_metrics_report.txt")
print("\nLaTeX table:")
print(table_path)
