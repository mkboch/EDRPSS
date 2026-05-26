from pathlib import Path
import pandas as pd
import json
import re
import urllib.request

ROOT = Path(r"G:\My Drive\1. Studies\RPI\Thesis\1. Prof Ge Wang\3. Extra Papers\3. MoltBot\Paper\OpenClaw_V3")
IN_CSV = ROOT / "results" / "llm_semantic_labeling" / "llm_labels_400_qwen25_7b.csv"
OUT_CSV = ROOT / "results" / "llm_semantic_labeling" / "llm_labels_400_qwen25_7b_repaired.csv"
OUT_RAW = ROOT / "results" / "llm_semantic_labeling" / "llm_labels_400_qwen25_7b_repair_raw.jsonl"

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

VALID_LABELS = {"corrective", "affirmation", "adversarial", "neutral", "unclear"}
VALID_BINARY = {"yes", "no", "unclear"}

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

def call_ollama(prompt, timeout=180):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 120
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
    text = text.strip()
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

def prompt_for_repair(comment_text):
    comment_text = str(comment_text)
    if len(comment_text) > 2500:
        comment_text = comment_text[:2500] + " ... [truncated]"

    return f"""Classify this single comment.

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

df = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False)

print("=" * 100)
print("REPAIR LLM PARSE ERRORS")
print("=" * 100)
print("Input:", IN_CSV)
print("Rows:", len(df))
print("Initial five-way distribution:")
print(df["llm_five_way_label"].value_counts(dropna=False))

# First normalize existing labels in case parse_error is recoverable from variants
df["llm_five_way_label"] = df["llm_five_way_label"].apply(normalize_label)
df["llm_is_corrective"] = df["llm_is_corrective"].apply(normalize_binary)

todo = df[df["llm_five_way_label"].isin(["parse_error", "error"]) | df["llm_is_corrective"].isin(["parse_error", "error"])].copy()
print("\nRows needing retry:", len(todo))

for n, idx in enumerate(todo.index, start=1):
    comment = df.loc[idx, "comment_text"]
    response = call_ollama(prompt_for_repair(comment))
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
    df.loc[idx, "parse_ok"] = (label not in {"parse_error", "error"} and binary not in {"parse_error", "error"})

    with open(OUT_RAW, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "annotation_id": df.loc[idx, "annotation_id"],
            "response": response,
            "parsed": obj,
            "final_label": label,
            "final_binary": binary
        }, ensure_ascii=False) + "\n")

    print(f"repaired {n}/{len(todo)} | annotation_id={df.loc[idx, 'annotation_id']} | label={label} | binary={binary}")

df["parse_ok"] = ~(
    df["llm_five_way_label"].isin(["parse_error", "error"]) |
    df["llm_is_corrective"].isin(["parse_error", "error"])
)

df.to_csv(OUT_CSV, index=False, encoding="utf-8")

print("\nDONE")
print("Output:", OUT_CSV)
print("Parse OK:", int(df["parse_ok"].sum()), "/", len(df))
print("\nFinal five-way distribution:")
print(df["llm_five_way_label"].value_counts(dropna=False))
print("\nFinal binary corrective distribution:")
print(df["llm_is_corrective"].value_counts(dropna=False))
