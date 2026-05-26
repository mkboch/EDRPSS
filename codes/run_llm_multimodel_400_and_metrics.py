from pathlib import Path
import pandas as pd
import json
import re
import urllib.request
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score

ROOT = Path(r"G:\My Drive\1. Studies\RPI\Thesis\1. Prof Ge Wang\3. Extra Papers\3. MoltBot\Paper\OpenClaw_V3")
HUMAN_DIR = ROOT / "Sample Dataset for Manual Validation" / "human_validation"
KEY_FILE = HUMAN_DIR / "human_validation_sample_400_KEY_DO_NOT_SHARE.csv"
A1_FILE = HUMAN_DIR / "human_validation_sample_400_ANNOTATED_annotator1.csv"
A2_FILE = HUMAN_DIR / "human_validation_sample_400_ANNOTATED_annotator2.csv"

OUTDIR = ROOT / "results" / "llm_semantic_labeling"
OUTDIR.mkdir(parents=True, exist_ok=True)
TEXDIR = ROOT / "latex_tables"
TEXDIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen2.5:7b", "qwen3:8b", "qwen3:4b", "qwen3:1.7b"]
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

VALID_LABELS = {"corrective", "affirmation", "adversarial", "neutral", "unclear"}
VALID_BINARY = {"yes", "no", "unclear"}
LABELS = ["corrective", "affirmation", "adversarial", "neutral"]
BINARY = ["yes", "no"]

LABEL_MAP = {
    "corrective signaling": "corrective",
    "correction": "corrective",
    "correction-like": "corrective",
    "corrective response": "corrective",
    "affirmative": "affirmation",
    "affirming": "affirmation",
    "supportive": "affirmation",
    "positive": "affirmation",
    "adversarial response": "adversarial",
    "hostile": "adversarial",
    "aggressive": "adversarial",
    "other": "neutral",
    "none": "neutral",
    "ordinary": "neutral",
    "ambiguous": "unclear",
    "unknown": "unclear",
}

def safe_model_name(model):
    return model.replace(":", "_").replace("/", "_").replace(".", "")

def norm_text(x):
    return str(x).strip().lower()

def normalize_label(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z\- ]", "", x).strip()
    if x in VALID_LABELS:
        return x
    return LABEL_MAP.get(x, "parse_error")

def normalize_binary(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z]", "", x)
    if x in VALID_BINARY:
        return x
    if x in {"true", "corrective"}:
        return "yes"
    if x in {"false", "notcorrective", "noncorrective"}:
        return "no"
    return "parse_error"

def call_ollama(prompt, model, timeout=180):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 140
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8", errors="replace"))
    return out.get("response", "")

def extract_json(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def make_prompt(comment_text):
    comment_text = str(comment_text)
    if len(comment_text) > 2500:
        comment_text = comment_text[:2500] + " ... [truncated]"

    return f"""Classify this single comment from an agent-only social archive.

Choose exactly one five_way_label from:
corrective, affirmation, adversarial, neutral, unclear

Choose exactly one is_corrective from:
yes, no, unclear

Definitions:
corrective = corrects, warns, cautions, criticizes, discourages, regulates, or pushes back.
affirmation = agrees, supports, praises, thanks, welcomes, validates, or encourages.
adversarial = hostile, insulting, threatening, provocative, manipulative, or aggressive.
neutral = ordinary, descriptive, informational, joking, or no clear category.
unclear = cannot decide from the comment alone.

Return only JSON. No explanation outside JSON.

Format:
{{"five_way_label":"neutral","is_corrective":"no","confidence":"medium","brief_reason":"short phrase"}}

Comment:
\"\"\"{comment_text}\"\"\"
"""

def label_one_model(model, base_df):
    tag = safe_model_name(model)
    out_csv = OUTDIR / f"llm_labels_400_{tag}.csv"
    raw_jsonl = OUTDIR / f"llm_labels_400_{tag}_raw.jsonl"

    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
        for _, r in prev.iterrows():
            done[str(r["annotation_id"])] = r.to_dict()
        print(f"[{model}] resume: {len(done)} rows already exist")

    rows = []
    for idx, r in base_df.iterrows():
        aid = str(r["annotation_id"])
        if aid in done:
            rows.append(done[aid])
            continue

        prompt = make_prompt(r["comment_text"])
        try:
            response = call_ollama(prompt, model=model)
            obj = extract_json(response)
            if obj is None:
                label = "parse_error"
                binary = "parse_error"
                conf = "parse_error"
                reason = response[:200].replace("\n", " ")
                parse_ok = False
            else:
                label = normalize_label(obj.get("five_way_label", ""))
                binary = normalize_binary(obj.get("is_corrective", ""))
                conf = str(obj.get("confidence", "")).strip().lower()
                reason = str(obj.get("brief_reason", "")).strip().replace("\n", " ")
                parse_ok = label != "parse_error" and binary != "parse_error"

            outrow = {
                "annotation_id": aid,
                "comment_id": r.get("comment_id", ""),
                "machine_label": r.get("machine_label", ""),
                "comment_text": r.get("comment_text", ""),
                "human_label_annotator1": r.get("human_label_annotator1", ""),
                "is_corrective_annotator1": r.get("is_corrective_annotator1", ""),
                "human_label_annotator2": r.get("human_label_annotator2", ""),
                "is_corrective_annotator2": r.get("is_corrective_annotator2", ""),
                "llm_model": model,
                "llm_five_way_label": label,
                "llm_is_corrective": binary,
                "llm_confidence": conf,
                "llm_reason": reason,
                "parse_ok": parse_ok,
            }

            with open(raw_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "annotation_id": aid,
                    "comment_id": r.get("comment_id", ""),
                    "response": response,
                    "parsed": obj,
                }, ensure_ascii=False) + "\n")

        except Exception as e:
            outrow = {
                "annotation_id": aid,
                "comment_id": r.get("comment_id", ""),
                "machine_label": r.get("machine_label", ""),
                "comment_text": r.get("comment_text", ""),
                "human_label_annotator1": r.get("human_label_annotator1", ""),
                "is_corrective_annotator1": r.get("is_corrective_annotator1", ""),
                "human_label_annotator2": r.get("human_label_annotator2", ""),
                "is_corrective_annotator2": r.get("is_corrective_annotator2", ""),
                "llm_model": model,
                "llm_five_way_label": "error",
                "llm_is_corrective": "error",
                "llm_confidence": "error",
                "llm_reason": repr(e),
                "parse_ok": False,
            }

        rows.append(outrow)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

        if (idx + 1) % 20 == 0:
            print(f"[{model}] processed {idx+1}/{len(base_df)}")

    df = pd.DataFrame(rows)

    # Repair any parse errors with one retry
    bad = df[
        df["llm_five_way_label"].isin(["parse_error", "error"]) |
        df["llm_is_corrective"].isin(["parse_error", "error"])
    ].copy()

    print(f"[{model}] initial parse OK: {len(df)-len(bad)}/{len(df)}; repair needed: {len(bad)}")

    for n, idx in enumerate(bad.index, start=1):
        prompt = make_prompt(df.loc[idx, "comment_text"])
        try:
            response = call_ollama(prompt, model=model)
            obj = extract_json(response)
            if obj is not None:
                label = normalize_label(obj.get("five_way_label", ""))
                binary = normalize_binary(obj.get("is_corrective", ""))
                conf = str(obj.get("confidence", "")).strip().lower()
                reason = str(obj.get("brief_reason", "")).strip().replace("\n", " ")
            else:
                label = "parse_error"
                binary = "parse_error"
                conf = "parse_error"
                reason = response[:200].replace("\n", " ")

            df.loc[idx, "llm_five_way_label"] = label
            df.loc[idx, "llm_is_corrective"] = binary
            df.loc[idx, "llm_confidence"] = conf
            df.loc[idx, "llm_reason"] = reason

            print(f"[{model}] repair {n}/{len(bad)} | label={label} | binary={binary}")
        except Exception as e:
            print(f"[{model}] repair failed {n}/{len(bad)}: {repr(e)}")

    df["parse_ok"] = ~(
        df["llm_five_way_label"].isin(["parse_error", "error"]) |
        df["llm_is_corrective"].isin(["parse_error", "error"])
    )

    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"\n[{model}] DONE")
    print(f"[{model}] Parse OK: {int(df['parse_ok'].sum())}/{len(df)}")
    print(f"[{model}] five-way distribution:")
    print(df["llm_five_way_label"].value_counts(dropna=False))
    print(f"[{model}] binary distribution:")
    print(df["llm_is_corrective"].value_counts(dropna=False))
    print(f"[{model}] saved: {out_csv}")

    return df

def binary_metrics(y_true, y_pred, reference_name, system_name):
    y_true = pd.Series(y_true).map(norm_text)
    y_pred = pd.Series(y_pred).map(norm_text)
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

def fmt3(x):
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"

print("=" * 100)
print("MULTI-MODEL LLM SEMANTIC LABELING ON 400 HUMAN-VALIDATION SAMPLE")
print("=" * 100)

key = pd.read_csv(KEY_FILE, dtype=str, keep_default_na=False)
a1 = pd.read_csv(A1_FILE, dtype=str, keep_default_na=False)
a2 = pd.read_csv(A2_FILE, dtype=str, keep_default_na=False)

base = key.merge(
    a1[["annotation_id", "human_label_annotator1", "is_corrective_annotator1"]],
    on="annotation_id",
    how="left"
).merge(
    a2[["annotation_id", "human_label_annotator2", "is_corrective_annotator2"]],
    on="annotation_id",
    how="left"
)

# Run labeling for all models. Existing completed CSVs are reused/resumed.
all_model_dfs = {}
for model in MODELS:
    print("\n" + "#" * 100)
    print("MODEL:", model)
    print("#" * 100)
    all_model_dfs[model] = label_one_model(model, base)

# Metrics
metric_rows = []

for model, df in all_model_dfs.items():
    df = df.copy()
    for c in df.columns:
        if c.startswith(("human_label", "is_corrective", "llm_", "machine_label")):
            df[c] = df[c].map(norm_text)

    df["machine_is_corrective"] = np.where(df["machine_label"] == "corrective", "yes", "no")

    df["human_binary_agree"] = np.where(
        (df["is_corrective_annotator1"].isin(BINARY)) &
        (df["is_corrective_annotator2"].isin(BINARY)) &
        (df["is_corrective_annotator1"] == df["is_corrective_annotator2"]),
        df["is_corrective_annotator1"],
        ""
    )

    metric_rows.append(binary_metrics(df["is_corrective_annotator1"], df["llm_is_corrective"], "Annotator 1", model))
    metric_rows.append(binary_metrics(df["is_corrective_annotator2"], df["llm_is_corrective"], "Annotator 2", model))
    metric_rows.append(binary_metrics(df["human_binary_agree"], df["llm_is_corrective"], "Agreement-only humans", model))
    metric_rows.append(binary_metrics(df["machine_is_corrective"], df["llm_is_corrective"], "Deterministic", model))

# Include deterministic baseline once using qwen2.5 df
df0 = all_model_dfs[MODELS[0]].copy()
for c in df0.columns:
    if c.startswith(("human_label", "is_corrective", "machine_label")):
        df0[c] = df0[c].map(norm_text)
df0["machine_is_corrective"] = np.where(df0["machine_label"] == "corrective", "yes", "no")
df0["human_binary_agree"] = np.where(
    (df0["is_corrective_annotator1"].isin(BINARY)) &
    (df0["is_corrective_annotator2"].isin(BINARY)) &
    (df0["is_corrective_annotator1"] == df0["is_corrective_annotator2"]),
    df0["is_corrective_annotator1"],
    ""
)
metric_rows.insert(0, binary_metrics(df0["is_corrective_annotator1"], df0["machine_is_corrective"], "Annotator 1", "Deterministic"))
metric_rows.insert(1, binary_metrics(df0["is_corrective_annotator2"], df0["machine_is_corrective"], "Annotator 2", "Deterministic"))
metric_rows.insert(2, binary_metrics(df0["human_binary_agree"], df0["machine_is_corrective"], "Agreement-only humans", "Deterministic"))

metrics = pd.DataFrame(metric_rows)
metrics_path = OUTDIR / "llm_multimodel_binary_metrics.csv"
metrics.to_csv(metrics_path, index=False)

# Compact main-table version: agreement-only humans only
compact = metrics[metrics["reference"] == "Agreement-only humans"].copy()
compact_path = OUTDIR / "llm_multimodel_agreement_only_metrics.csv"
compact.to_csv(compact_path, index=False)

tex_path = TEXDIR / "table_llm_multimodel_semantic_validation.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{Multi-model LLM-assisted semantic calibration on the 400-comment validation sample.} Binary correction-like metrics are computed against the agreement-only human subset.}
\label{tab:llm_multimodel_semantic_validation}
\begin{tabular}{@{}lrrrr@{}}
\toprule
System & Precision & Recall & F1 & Accuracy \\
\midrule
""")
    for _, r in compact.iterrows():
        f.write(
            f"{r['system']} & {fmt3(r['precision'])} & {fmt3(r['recall'])} & {fmt3(r['f1'])} & {fmt3(r['accuracy'])} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

report_path = OUTDIR / "llm_multimodel_validation_report.txt"
lines = []
lines.append("=" * 100)
lines.append("MULTI-MODEL LLM SEMANTIC VALIDATION REPORT")
lines.append("=" * 100)
lines.append("Models: " + ", ".join(MODELS))
lines.append("")
lines.append("All binary metrics:")
lines.append(metrics.to_string(index=False))
lines.append("")
lines.append("Agreement-only human compact metrics:")
lines.append(compact.to_string(index=False))
lines.append("")
lines.append("Outputs:")
lines.append(str(metrics_path))
lines.append(str(compact_path))
lines.append(str(tex_path))
report_path.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "\n".join(lines))
