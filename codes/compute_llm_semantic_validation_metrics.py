from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix

ROOT = Path(r"G:\My Drive\1. Studies\RPI\Thesis\1. Prof Ge Wang\3. Extra Papers\3. MoltBot\Paper\OpenClaw_V3")

IN_CSV = ROOT / "results" / "llm_semantic_labeling" / "llm_labels_400_qwen25_7b_repaired.csv"
OUTDIR = ROOT / "results" / "llm_semantic_labeling"
OUT_REPORT = OUTDIR / "llm_semantic_validation_report.txt"
OUT_BINARY = OUTDIR / "llm_binary_corrective_metrics.csv"
OUT_MULTI = OUTDIR / "llm_five_way_metrics.csv"
OUT_CROSSTAB = OUTDIR / "llm_human_crosstabs.txt"
OUT_TEX = ROOT / "latex_tables" / "table_llm_semantic_validation.tex"
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

LABELS = ["corrective", "affirmation", "adversarial", "neutral"]
BINARY = ["yes", "no"]

def norm(x):
    return str(x).strip().lower()

def binary_metrics(y_true, y_pred, reference_name, system_name):
    y_true = pd.Series(y_true).map(norm)
    y_pred = pd.Series(y_pred).map(norm)
    mask = y_true.isin(BINARY) & y_pred.isin(BINARY)
    yt = y_true[mask]
    yp = y_pred[mask]

    p, r, f1, _ = precision_recall_fscore_support(
        yt == "yes",
        yp == "yes",
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(yt, yp)
    kappa = cohen_kappa_score(yt, yp)

    tp = int(((yt == "yes") & (yp == "yes")).sum())
    fp = int(((yt == "no") & (yp == "yes")).sum())
    fn = int(((yt == "yes") & (yp == "no")).sum())
    tn = int(((yt == "no") & (yp == "no")).sum())

    return {
        "reference": reference_name,
        "system": system_name,
        "N": int(mask.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": acc,
        "kappa": kappa,
    }

def multiclass_metrics(y_true, y_pred, reference_name, system_name):
    y_true = pd.Series(y_true).map(norm)
    y_pred = pd.Series(y_pred).map(norm)
    mask = y_true.isin(LABELS) & y_pred.isin(LABELS)
    yt = y_true[mask]
    yp = y_pred[mask]

    rows = []
    p, r, f1, support = precision_recall_fscore_support(
        yt,
        yp,
        labels=LABELS,
        zero_division=0,
    )
    for label, pp, rr, ff, ss in zip(LABELS, p, r, f1, support):
        rows.append({
            "reference": reference_name,
            "system": system_name,
            "label": label,
            "support": int(ss),
            "precision": pp,
            "recall": rr,
            "f1": ff,
        })

    rows.append({
        "reference": reference_name,
        "system": system_name,
        "label": "overall_accuracy",
        "support": int(mask.sum()),
        "precision": np.nan,
        "recall": np.nan,
        "f1": accuracy_score(yt, yp),
    })

    rows.append({
        "reference": reference_name,
        "system": system_name,
        "label": "cohen_kappa",
        "support": int(mask.sum()),
        "precision": np.nan,
        "recall": np.nan,
        "f1": cohen_kappa_score(yt, yp),
    })
    return rows

def fmt3(x):
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"

df = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False)

# Normalize columns
for c in df.columns:
    if c.startswith(("human_label", "is_corrective", "llm_", "machine_label")):
        df[c] = df[c].map(norm)

# Deterministic binary from machine label
df["machine_is_corrective"] = np.where(df["machine_label"] == "corrective", "yes", "no")

# Human agreement-only binary reference
df["human_binary_agree"] = np.where(
    (df["is_corrective_annotator1"].isin(BINARY)) &
    (df["is_corrective_annotator2"].isin(BINARY)) &
    (df["is_corrective_annotator1"] == df["is_corrective_annotator2"]),
    df["is_corrective_annotator1"],
    ""
)

# Human agreement-only five-way reference
df["human_label_agree"] = np.where(
    (df["human_label_annotator1"].isin(LABELS)) &
    (df["human_label_annotator2"].isin(LABELS)) &
    (df["human_label_annotator1"] == df["human_label_annotator2"]),
    df["human_label_annotator1"],
    ""
)

binary_rows = []
binary_rows.append(binary_metrics(df["is_corrective_annotator1"], df["machine_is_corrective"], "Annotator 1", "Deterministic"))
binary_rows.append(binary_metrics(df["is_corrective_annotator2"], df["machine_is_corrective"], "Annotator 2", "Deterministic"))
binary_rows.append(binary_metrics(df["human_binary_agree"], df["machine_is_corrective"], "Agreement-only humans", "Deterministic"))

binary_rows.append(binary_metrics(df["is_corrective_annotator1"], df["llm_is_corrective"], "Annotator 1", "Qwen2.5-7B"))
binary_rows.append(binary_metrics(df["is_corrective_annotator2"], df["llm_is_corrective"], "Annotator 2", "Qwen2.5-7B"))
binary_rows.append(binary_metrics(df["human_binary_agree"], df["llm_is_corrective"], "Agreement-only humans", "Qwen2.5-7B"))

binary_rows.append(binary_metrics(df["machine_is_corrective"], df["llm_is_corrective"], "Deterministic", "Qwen2.5-7B"))

binary_df = pd.DataFrame(binary_rows)
binary_df.to_csv(OUT_BINARY, index=False)

multi_rows = []
multi_rows.extend(multiclass_metrics(df["human_label_annotator1"], df["machine_label"], "Annotator 1", "Deterministic"))
multi_rows.extend(multiclass_metrics(df["human_label_annotator2"], df["machine_label"], "Annotator 2", "Deterministic"))
multi_rows.extend(multiclass_metrics(df["human_label_agree"], df["machine_label"], "Agreement-only humans", "Deterministic"))

multi_rows.extend(multiclass_metrics(df["human_label_annotator1"], df["llm_five_way_label"], "Annotator 1", "Qwen2.5-7B"))
multi_rows.extend(multiclass_metrics(df["human_label_annotator2"], df["llm_five_way_label"], "Annotator 2", "Qwen2.5-7B"))
multi_rows.extend(multiclass_metrics(df["human_label_agree"], df["llm_five_way_label"], "Agreement-only humans", "Qwen2.5-7B"))

multi_df = pd.DataFrame(multi_rows)
multi_df.to_csv(OUT_MULTI, index=False)

# Crosstabs
ct_lines = []
ct_lines.append("=" * 100)
ct_lines.append("LLM SEMANTIC LABELING CROSSTABS")
ct_lines.append("=" * 100)
for ref_col, ref_name in [
    ("human_label_annotator1", "Annotator 1 five-way"),
    ("human_label_annotator2", "Annotator 2 five-way"),
    ("machine_label", "Deterministic five-way"),
]:
    ct_lines.append("\n" + "-" * 100)
    ct_lines.append(f"{ref_name} vs LLM five-way")
    ct_lines.append(pd.crosstab(df[ref_col], df["llm_five_way_label"]).to_string())

for ref_col, ref_name in [
    ("is_corrective_annotator1", "Annotator 1 binary"),
    ("is_corrective_annotator2", "Annotator 2 binary"),
    ("machine_is_corrective", "Deterministic binary"),
]:
    ct_lines.append("\n" + "-" * 100)
    ct_lines.append(f"{ref_name} vs LLM binary")
    ct_lines.append(pd.crosstab(df[ref_col], df["llm_is_corrective"]).to_string())

OUT_CROSSTAB.write_text("\n".join(ct_lines), encoding="utf-8")

# Compact LaTeX table: binary corrective only
rows_for_tex = binary_df[
    binary_df["reference"].isin(["Annotator 1", "Annotator 2", "Agreement-only humans"]) &
    binary_df["system"].isin(["Deterministic", "Qwen2.5-7B"])
].copy()

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{LLM-assisted semantic validation on the 400-comment human-validation sample.} Binary correction-like metrics compare deterministic labels and Qwen2.5-7B labels against human annotators.}
\label{tab:llm_semantic_validation}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Reference & System & Precision & Recall & F1 & Accuracy \\
\midrule
""")
    for _, r in rows_for_tex.iterrows():
        f.write(
            f"{r['reference']} & {r['system']} & "
            f"{fmt3(r['precision'])} & {fmt3(r['recall'])} & "
            f"{fmt3(r['f1'])} & {fmt3(r['accuracy'])} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

# Text report
lines = []
lines.append("=" * 100)
lines.append("LLM SEMANTIC VALIDATION REPORT")
lines.append("=" * 100)
lines.append(f"Input: {IN_CSV}")
lines.append(f"Rows: {len(df)}")
lines.append("")
lines.append("Label distributions:")
lines.append("Deterministic machine_label:")
lines.append(df["machine_label"].value_counts().to_string())
lines.append("")
lines.append("LLM five-way:")
lines.append(df["llm_five_way_label"].value_counts().to_string())
lines.append("")
lines.append("LLM binary:")
lines.append(df["llm_is_corrective"].value_counts().to_string())
lines.append("")
lines.append("BINARY CORRECTIVE METRICS:")
lines.append(binary_df.to_string(index=False))
lines.append("")
lines.append("MULTICLASS FIVE-WAY METRICS:")
lines.append(multi_df.to_string(index=False))
lines.append("")
lines.append("Saved:")
lines.append(str(OUT_BINARY))
lines.append(str(OUT_MULTI))
lines.append(str(OUT_CROSSTAB))
lines.append(str(OUT_TEX))

report = "\n".join(lines)
OUT_REPORT.write_text(report, encoding="utf-8")

print(report)
