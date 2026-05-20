from pathlib import Path
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
OUT = ROOT / "journal_upgrade_outputs_round4_auto_audit"
OUT.mkdir(exist_ok=True)

COMMENTS_FILE = ROOT / "Datasets" / "comments_labeled.csv"
POSTS_FILE = ROOT / "results" / "posts_with_di.csv"

print("Project root:", ROOT)
print("Using comments:", COMMENTS_FILE)
print("Using posts:", POSTS_FILE)
print("Output folder:", OUT)

# -----------------------------
# Load comments and posts
# -----------------------------
comment_cols = [
    "id", "post_id", "agent_id", "agent_name", "content",
    "score", "created_at", "dump_date", "response_type", "response_type_legacy", "di_comment"
]

post_cols = [
    "id", "agent_id", "agent_name", "submolt", "title", "content",
    "score", "comment_count", "created_at", "dump_date", "di_post"
]

comments = pd.read_csv(COMMENTS_FILE, usecols=lambda c: c in comment_cols, low_memory=False)
posts = pd.read_csv(POSTS_FILE, usecols=lambda c: c in post_cols, low_memory=False)

comments["id"] = comments["id"].astype(str)
comments["post_id"] = comments["post_id"].astype(str)
comments["content"] = comments["content"].fillna("").astype(str)
comments["response_type"] = comments["response_type"].fillna("neutral").astype(str).str.lower()

posts["id"] = posts["id"].astype(str)
posts["post_text"] = posts["title"].fillna("").astype(str) + "\n\n" + posts["content"].fillna("").astype(str)
posts["post_chars"] = posts["post_text"].str.len()
posts["post_words"] = posts["post_text"].str.split().str.len()

post_lookup = posts[[
    "id", "agent_id", "agent_name", "submolt", "di_post",
    "score", "comment_count", "dump_date", "post_text", "post_chars", "post_words"
]].rename(columns={
    "id": "post_id",
    "agent_id": "post_author_id",
    "agent_name": "post_author_name",
    "score": "post_score",
    "dump_date": "post_dump_date",
})

df = comments.merge(post_lookup, on="post_id", how="left")
df["comment_chars"] = df["content"].str.len()
df["comment_words"] = df["content"].str.split().str.len()

print("Rows:", len(df))
print("Response type counts:")
print(df["response_type"].value_counts())

# -----------------------------
# Automatic response-rule audit patterns
# These are not changing your labels.
# They are only post-hoc diagnostics for interpretability.
# -----------------------------
pattern_groups = {
    "corrective_warning": [
        r"\b(be careful|careful|caution|warning|warn|risky|risk|danger|unsafe|harmful)\b",
        r"\b(should not|shouldn't|do not|don't|cannot|can't|must not|avoid|stop)\b",
        r"\b(not appropriate|inappropriate|not allowed|violate|violation)\b",
    ],
    "corrective_disagreement_or_correction": [
        r"\b(i disagree|disagree|not correct|incorrect|wrong|false|misleading|mistaken)\b",
        r"\b(actually|instead|rather|correction|correcting|clarify|clarification)\b",
        r"\b(this is not|that is not|it is not|not true)\b",
    ],
    "corrective_refusal_or_boundary": [
        r"\b(i can'?t help|cannot help|can't assist|cannot assist|won't help|will not help)\b",
        r"\b(i cannot|i can't|unable to|not able to)\b",
    ],
    "affirmation_agreement": [
        r"\b(agree|i agree|yes|yeah|yep|correct|right|exactly|true|good point)\b",
        r"\b(great|nice|excellent|well said|makes sense|sounds good)\b",
    ],
    "adversarial_hostile": [
        r"\b(stupid|idiot|dumb|nonsense|trash|garbage|ridiculous|pathetic)\b",
        r"\b(shut up|you are wrong|you're wrong|liar|lying)\b",
    ],
    "neutral_question_or_info": [
        r"\b(what|why|how|when|where|which|could you|can you|would you)\b",
        r"\b(information|explain|describe|details|example|summary)\b",
    ],
}

compiled = {
    group: [re.compile(p, flags=re.IGNORECASE) for p in pats]
    for group, pats in pattern_groups.items()
}

def matched_groups(text):
    out = []
    for group, pats in compiled.items():
        for pat in pats:
            m = pat.search(text)
            if m:
                out.append((group, m.group(0)))
                break
    return out

def first_match_for_group(text, group):
    for pat in compiled[group]:
        m = pat.search(text)
        if m:
            return m.group(0)
    return ""

all_groups = list(pattern_groups.keys())

for group in all_groups:
    df[f"audit_match_{group}"] = df["content"].map(lambda x, g=group: bool(first_match_for_group(x, g)))
    df[f"audit_phrase_{group}"] = df["content"].map(lambda x, g=group: first_match_for_group(x, g))

df["audit_n_groups_matched"] = df[[f"audit_match_{g}" for g in all_groups]].sum(axis=1)

# broad group flags
df["audit_any_corrective_pattern"] = df[
    ["audit_match_corrective_warning", "audit_match_corrective_disagreement_or_correction", "audit_match_corrective_refusal_or_boundary"]
].any(axis=1)

df["audit_any_affirmation_pattern"] = df[
    ["audit_match_affirmation_agreement"]
].any(axis=1)

df["audit_any_adversarial_pattern"] = df[
    ["audit_match_adversarial_hostile"]
].any(axis=1)

df["audit_any_neutral_pattern"] = df[
    ["audit_match_neutral_question_or_info"]
].any(axis=1)

# -----------------------------
# 1. Pattern coverage by assigned response type
# -----------------------------
coverage_rows = []
for label, sub in df.groupby("response_type"):
    row = {
        "response_type": label,
        "n": len(sub),
        "mean_comment_words": sub["comment_words"].mean(),
        "mean_comment_chars": sub["comment_chars"].mean(),
        "mean_di_comment": pd.to_numeric(sub.get("di_comment", np.nan), errors="coerce").mean(),
    }
    for group in all_groups:
        row[f"pct_match_{group}"] = sub[f"audit_match_{group}"].mean()
    row["pct_any_corrective_pattern"] = sub["audit_any_corrective_pattern"].mean()
    row["pct_any_affirmation_pattern"] = sub["audit_any_affirmation_pattern"].mean()
    row["pct_any_adversarial_pattern"] = sub["audit_any_adversarial_pattern"].mean()
    row["pct_any_neutral_pattern"] = sub["audit_any_neutral_pattern"].mean()
    row["pct_multiple_groups_matched"] = (sub["audit_n_groups_matched"] >= 2).mean()
    row["pct_no_audit_pattern_matched"] = (sub["audit_n_groups_matched"] == 0).mean()
    coverage_rows.append(row)

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(OUT / "round4_pattern_coverage_by_response_type.csv", index=False)

# -----------------------------
# 2. Confusion-style diagnostic using broad patterns
# Not true ground truth. Just lexical consistency check.
# -----------------------------
def audit_pred_label(row):
    # precedence similar to a deterministic labeling diagnostic
    if row["audit_any_corrective_pattern"]:
        return "corrective_like"
    if row["audit_any_adversarial_pattern"]:
        return "adversarial_like"
    if row["audit_any_affirmation_pattern"]:
        return "affirmation_like"
    if row["audit_any_neutral_pattern"]:
        return "neutral_like"
    return "no_pattern"

df["audit_predicted_label_family"] = df.apply(audit_pred_label, axis=1)

confusion = pd.crosstab(
    df["response_type"],
    df["audit_predicted_label_family"],
    margins=True
)

confusion.to_csv(OUT / "round4_audit_pattern_confusion_table.csv")

# -----------------------------
# 3. Representative examples by label and audit pattern
# -----------------------------
def clean_excerpt(s, n=500):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:n]

example_rows = []

for label in ["corrective", "affirmation", "adversarial", "neutral"]:
    sub_label = df[df["response_type"] == label].copy()
    if len(sub_label) == 0:
        continue

    # examples with expected broad pattern
    if label == "corrective":
        expected_col = "audit_any_corrective_pattern"
    elif label == "affirmation":
        expected_col = "audit_any_affirmation_pattern"
    elif label == "adversarial":
        expected_col = "audit_any_adversarial_pattern"
    else:
        expected_col = "audit_any_neutral_pattern"

    for status_name, status_df in [
        ("expected_pattern_present", sub_label[sub_label[expected_col]]),
        ("no_expected_pattern", sub_label[~sub_label[expected_col]]),
        ("multiple_pattern_groups", sub_label[sub_label["audit_n_groups_matched"] >= 2]),
    ]:
        if len(status_df) == 0:
            continue

        sample_n = min(25, len(status_df))
        sample = status_df.sample(n=sample_n, random_state=42)

        for _, r in sample.iterrows():
            matched = []
            phrases = []
            for group in all_groups:
                if r[f"audit_match_{group}"]:
                    matched.append(group)
                    phrases.append(f"{group}: {r[f'audit_phrase_{group}']}")

            example_rows.append({
                "response_type": label,
                "example_status": status_name,
                "audit_predicted_label_family": r["audit_predicted_label_family"],
                "matched_groups": "; ".join(matched),
                "matched_phrases": "; ".join(phrases),
                "comment_id": r["id"],
                "post_id": r["post_id"],
                "di_comment": r.get("di_comment", ""),
                "di_post": r.get("di_post", ""),
                "submolt": r.get("submolt", ""),
                "comment_excerpt": clean_excerpt(r["content"], 700),
                "post_excerpt": clean_excerpt(r.get("post_text", ""), 700),
            })

examples = pd.DataFrame(example_rows)
examples.to_csv(OUT / "round4_representative_label_examples.csv", index=False)

# -----------------------------
# 4. Short examples table for paper
# choose compact examples where label and audit pattern align
# -----------------------------
paper_examples = []

paper_targets = [
    ("corrective", "audit_any_corrective_pattern"),
    ("affirmation", "audit_any_affirmation_pattern"),
    ("adversarial", "audit_any_adversarial_pattern"),
    ("neutral", "audit_any_neutral_pattern"),
]

for label, col in paper_targets:
    sub = df[(df["response_type"] == label) & (df[col])].copy()
    if len(sub) == 0:
        sub = df[df["response_type"] == label].copy()
    sub = sub.sort_values(["comment_words", "comment_chars"], ascending=[True, True])
    sub = sub[(sub["comment_words"] >= 5) & (sub["comment_words"] <= 60)]
    if len(sub) == 0:
        continue

    sample = sub.sample(n=min(8, len(sub)), random_state=7)
    for _, r in sample.iterrows():
        matched = []
        phrases = []
        for group in all_groups:
            if r[f"audit_match_{group}"]:
                matched.append(group)
                phrases.append(str(r[f"audit_phrase_{group}"]))

        paper_examples.append({
            "response_type": label,
            "matched_rule_family": "; ".join(matched),
            "matched_phrase": "; ".join(phrases),
            "comment_excerpt": clean_excerpt(r["content"], 350),
            "post_excerpt": clean_excerpt(r.get("post_text", ""), 350),
            "di_post": r.get("di_post", ""),
            "di_comment": r.get("di_comment", ""),
        })

paper_examples = pd.DataFrame(paper_examples)
paper_examples.to_csv(OUT / "round4_candidate_examples_for_paper_table.csv", index=False)

# -----------------------------
# 5. Automatic quality-risk summary
# -----------------------------
risk_rows = []

for label, sub in df.groupby("response_type"):
    if label == "corrective":
        expected = sub["audit_any_corrective_pattern"]
    elif label == "affirmation":
        expected = sub["audit_any_affirmation_pattern"]
    elif label == "adversarial":
        expected = sub["audit_any_adversarial_pattern"]
    elif label == "neutral":
        expected = sub["audit_any_neutral_pattern"]
    else:
        expected = pd.Series(False, index=sub.index)

    risk_rows.append({
        "response_type": label,
        "n": len(sub),
        "pct_expected_pattern_present": expected.mean(),
        "pct_expected_pattern_absent": (~expected).mean(),
        "pct_multiple_pattern_groups": (sub["audit_n_groups_matched"] >= 2).mean(),
        "pct_no_pattern_matched": (sub["audit_n_groups_matched"] == 0).mean(),
        "mean_comment_words": sub["comment_words"].mean(),
        "median_comment_words": sub["comment_words"].median(),
    })

risk = pd.DataFrame(risk_rows)
risk.to_csv(OUT / "round4_auto_label_quality_risk_summary.csv", index=False)

# -----------------------------
# 6. Write readable markdown report
# -----------------------------
md = []
md.append("# Round 4 Automatic Label Audit Report\n")
md.append("This report is an automatic lexical audit, not human validation. It checks whether assigned response labels align with broad diagnostic lexical patterns and identifies possible ambiguity risks.\n")

md.append("## Dataset summary\n")
md.append(f"- Total comments: {len(df):,}\n")
md.append(f"- Unique posts: {df['post_id'].nunique():,}\n")
md.append("\n### Response type counts\n")
md.append(df["response_type"].value_counts().to_markdown())

md.append("\n\n## Automatic quality-risk summary\n")
md.append(risk.to_markdown(index=False))

md.append("\n\n## Pattern coverage by response type\n")
show_cols = ["response_type", "n", "pct_any_corrective_pattern", "pct_any_affirmation_pattern", "pct_any_adversarial_pattern", "pct_any_neutral_pattern", "pct_multiple_groups_matched", "pct_no_audit_pattern_matched"]
md.append(coverage[show_cols].to_markdown(index=False))

md.append("\n\n## Diagnostic interpretation\n")
md.append("- High expected-pattern coverage supports lexical auditability of deterministic labels.\n")
md.append("- High expected-pattern absence suggests that labels may rely on rules not represented in this post-hoc audit or may include semantically ambiguous cases.\n")
md.append("- High multiple-pattern coverage indicates possible rule-precedence conflicts.\n")
md.append("- This audit cannot establish true pragmatic correctness without human annotation.\n")

md.append("\n\n## Files generated\n")
for p in sorted(OUT.glob("*")):
    md.append(f"- `{p.name}`")

(OUT / "round4_auto_audit_report.md").write_text("\n".join(md), encoding="utf-8")

# -----------------------------
# 7. Save machine-readable summary JSON
# -----------------------------
summary = {
    "n_comments": int(len(df)),
    "n_posts": int(df["post_id"].nunique()),
    "response_type_counts": df["response_type"].value_counts().to_dict(),
    "generated_files": [p.name for p in sorted(OUT.glob("*"))],
    "note": "Automatic lexical audit only; not human semantic validation."
}

(OUT / "round4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nDONE. Generated files:")
for p in sorted(OUT.glob("*")):
    print(" -", p)

print("\nMain risk summary:")
print(risk.to_string(index=False))
