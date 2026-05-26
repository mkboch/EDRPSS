from pathlib import Path
import pandas as pd
import json
import re
import time
import urllib.request
import urllib.error

ROOT = Path(r"G:\My Drive\1. Studies\RPI\Thesis\1. Prof Ge Wang\3. Extra Papers\3. MoltBot\Paper\OpenClaw_V3")
HUMAN_DIR = ROOT / "Sample Dataset for Manual Validation" / "human_validation"

KEY_FILE = HUMAN_DIR / "human_validation_sample_400_KEY_DO_NOT_SHARE.csv"
A1_FILE = HUMAN_DIR / "human_validation_sample_400_ANNOTATED_annotator1.csv"
A2_FILE = HUMAN_DIR / "human_validation_sample_400_ANNOTATED_annotator2.csv"

OUTDIR = ROOT / "results" / "llm_semantic_labeling"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "llm_labels_400_qwen25_7b.csv"
OUT_RAW = OUTDIR / "llm_labels_400_qwen25_7b_raw.jsonl"

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

VALID_LABELS = {"corrective", "affirmation", "adversarial", "neutral", "unclear"}
VALID_BINARY = {"yes", "no", "unclear"}

def call_ollama(prompt, model=MODEL, timeout=180):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 160
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
    # remove markdown fences if present
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()

    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # try first JSON object
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def norm_label(x):
    x = str(x).strip().lower()
    return x if x in VALID_LABELS else "parse_error"

def norm_binary(x):
    x = str(x).strip().lower()
    return x if x in VALID_BINARY else "parse_error"

def make_prompt(comment_text):
    # Keep the same visible comment-only setting as human annotation.
    comment_text = str(comment_text)
    if len(comment_text) > 3000:
        comment_text = comment_text[:3000] + " ... [truncated]"

    return f"""You are labeling one comment from an agent-only social archive.

Classify ONLY the comment text. Do not infer hidden intent. Do not use outside context.

Definitions:
- corrective: discourages, corrects, cautions, warns, regulates, criticizes, or pushes back against a behavior, claim, instruction, request, or action.
- affirmation: agrees with, supports, praises, validates, thanks, welcomes, or positively reinforces.
- adversarial: hostile, insulting, threatening, provocative, manipulative, or aggressive toward another agent or group.
- neutral: informational, descriptive, off-topic, joking, ambiguous, or ordinary interaction without clear correction, affirmation, or adversarial tone.
- unclear: impossible to decide from the visible comment alone.

Also answer whether the comment is correction-like:
- yes: clearly correction-like under the definition above.
- no: clearly not correction-like.
- unclear: uncertain.

Return ONLY valid JSON with exactly these keys:
{{
  "five_way_label": "corrective|affirmation|adversarial|neutral|unclear",
  "is_corrective": "yes|no|unclear",
  "confidence": "low|medium|high",
  "brief_reason": "short phrase, maximum 20 words"
}}

Comment:
\"\"\"{comment_text}\"\"\"
"""

print("=" * 100)
print("LLM SEMANTIC LABELING: 400 HUMAN-VALIDATION SAMPLE")
print("=" * 100)
print("Model:", MODEL)
print("Key file:", KEY_FILE)
print("Output:", OUT_CSV)

if not KEY_FILE.exists():
    raise FileNotFoundError(KEY_FILE)

key = pd.read_csv(KEY_FILE, dtype=str, keep_default_na=False)
a1 = pd.read_csv(A1_FILE, dtype=str, keep_default_na=False) if A1_FILE.exists() else None
a2 = pd.read_csv(A2_FILE, dtype=str, keep_default_na=False) if A2_FILE.exists() else None

df = key.copy()
if a1 is not None:
    df = df.merge(
        a1[["annotation_id", "human_label_annotator1", "is_corrective_annotator1"]],
        on="annotation_id",
        how="left"
    )
if a2 is not None:
    df = df.merge(
        a2[["annotation_id", "human_label_annotator2", "is_corrective_annotator2"]],
        on="annotation_id",
        how="left"
    )

# Resume support
done = {}
if OUT_CSV.exists():
    prev = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)
    for _, r in prev.iterrows():
        done[str(r["annotation_id"])] = r.to_dict()
    print("Resume mode: existing rows =", len(done))

rows = []
for idx, r in df.iterrows():
    aid = str(r["annotation_id"])

    if aid in done:
        rows.append(done[aid])
        continue

    prompt = make_prompt(r["comment_text"])

    try:
        response = call_ollama(prompt)
        obj = extract_json(response)

        if obj is None:
            llm_label = "parse_error"
            llm_binary = "parse_error"
            confidence = "parse_error"
            reason = response[:200].replace("\n", " ")
            parse_ok = False
        else:
            llm_label = norm_label(obj.get("five_way_label", ""))
            llm_binary = norm_binary(obj.get("is_corrective", ""))
            confidence = str(obj.get("confidence", "")).strip().lower()
            reason = str(obj.get("brief_reason", "")).strip().replace("\n", " ")
            parse_ok = (llm_label != "parse_error" and llm_binary != "parse_error")

        outrow = {
            "annotation_id": aid,
            "comment_id": r.get("comment_id", ""),
            "machine_label": r.get("machine_label", ""),
            "comment_text": r.get("comment_text", ""),
            "human_label_annotator1": r.get("human_label_annotator1", ""),
            "is_corrective_annotator1": r.get("is_corrective_annotator1", ""),
            "human_label_annotator2": r.get("human_label_annotator2", ""),
            "is_corrective_annotator2": r.get("is_corrective_annotator2", ""),
            "llm_model": MODEL,
            "llm_five_way_label": llm_label,
            "llm_is_corrective": llm_binary,
            "llm_confidence": confidence,
            "llm_reason": reason,
            "parse_ok": parse_ok,
        }

        with open(OUT_RAW, "a", encoding="utf-8") as f:
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
            "llm_model": MODEL,
            "llm_five_way_label": "error",
            "llm_is_corrective": "error",
            "llm_confidence": "error",
            "llm_reason": repr(e),
            "parse_ok": False,
        }

    rows.append(outrow)

    # Save every row so progress is not lost
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    if (idx + 1) % 20 == 0:
        print(f"Processed {idx+1}/{len(df)}")

print("\nDONE")
out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False, encoding="utf-8")

print("Rows:", len(out))
print("Parse OK:", out["parse_ok"].sum(), "/", len(out))
print("\nLLM five-way distribution:")
print(out["llm_five_way_label"].value_counts(dropna=False))
print("\nLLM binary corrective distribution:")
print(out["llm_is_corrective"].value_counts(dropna=False))
print("\nSaved:", OUT_CSV)
