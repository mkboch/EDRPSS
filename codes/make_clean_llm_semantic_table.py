from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"G:\My Drive\1. Studies\RPI\Thesis\1. Prof Ge Wang\3. Extra Papers\3. MoltBot\Paper\OpenClaw_V3")
IN_CSV = ROOT / "results" / "llm_semantic_labeling" / "llm_jsonmode_multimodel_agreement_only_metrics.csv"
OUT_CSV = ROOT / "results" / "llm_semantic_labeling" / "llm_valid_model_agreement_only_metrics.csv"
OUT_TEX = ROOT / "latex_tables" / "table_llm_multimodel_semantic_validation.tex"

df = pd.read_csv(IN_CSV)

keep = df["system"].isin(["Deterministic", "qwen2.5:7b", "qwen3:8b"])
df = df[keep].copy()

df.to_csv(OUT_CSV, index=False)

def fmt3(x):
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{LLM-assisted semantic calibration on the 400-comment validation sample.} Binary correction-like metrics are computed against the agreement-only human subset. Deterministic labels provide the scalable rule-based baseline; local LLM labels provide semantic calibration.}
\label{tab:llm_multimodel_semantic_validation}
\begin{tabular}{@{}lrrrr@{}}
\toprule
System & Precision & Recall & F1 & Accuracy \\
\midrule
""")
    for _, r in df.iterrows():
        f.write(
            f"{r['system']} & {fmt3(r['precision'])} & {fmt3(r['recall'])} & {fmt3(r['f1'])} & {fmt3(r['accuracy'])} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

print("Saved CSV:", OUT_CSV)
print("Saved LaTeX:", OUT_TEX)
print(df.to_string(index=False))
