from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(".").resolve()
OUT = ROOT / "journal_upgrade_outputs_round3"
OUT.mkdir(exist_ok=True)

POSTS_FILE = ROOT / "results" / "posts_with_di.csv"
COMMENTS_FILE = ROOT / "Datasets" / "comments_labeled.csv"

print("Project root:", ROOT)
print("Using posts:", POSTS_FILE)
print("Using comments:", COMMENTS_FILE)
print("Output folder:", OUT)

def safe_log1p(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return np.log1p(s.clip(lower=0))

def fit_cluster_logit(df, y_col, x_cols, cluster_col="post_id"):
    use = df[[y_col, cluster_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    print("Model rows:", len(use), "clusters:", use[cluster_col].nunique())
    y = use[y_col].astype(float)
    X = sm.add_constant(use[x_cols].astype(float), has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": use[cluster_col].astype(str)})

    rows = []
    for x in x_cols:
        beta = float(res.params[x])
        se = float(res.bse[x])
        rows.append({
            "term": x,
            "beta": beta,
            "se_cluster_post": se,
            "p_value": float(res.pvalues[x]),
            "odds_ratio": float(np.exp(beta)),
            "or_ci_low": float(np.exp(beta - 1.96 * se)),
            "or_ci_high": float(np.exp(beta + 1.96 * se)),
            "n_used": int(len(use)),
            "n_clusters": int(use[cluster_col].nunique()),
        })
    return rows

# Load needed columns only
post_cols = [
    "id", "agent_id", "agent_name", "submolt", "title", "content",
    "score", "comment_count", "created_at", "dump_date", "di_post"
]
comment_cols = [
    "id", "post_id", "agent_id", "agent_name", "content",
    "score", "created_at", "dump_date", "response_type", "di_comment"
]

posts = pd.read_csv(POSTS_FILE, usecols=lambda c: c in post_cols, low_memory=False)
comments = pd.read_csv(COMMENTS_FILE, usecols=lambda c: c in comment_cols, low_memory=False)

posts["id"] = posts["id"].astype(str)
comments["post_id"] = comments["post_id"].astype(str)

comments["response_type"] = comments["response_type"].fillna("neutral").astype(str).str.lower()
comments["is_corrective"] = (comments["response_type"] == "corrective").astype(int)

posts["di_post"] = pd.to_numeric(posts["di_post"], errors="coerce").fillna(0).clip(0, 10)
posts["di_binary"] = (posts["di_post"] > 0).astype(int)

posts["post_text"] = posts["title"].fillna("").astype(str) + "\n\n" + posts["content"].fillna("").astype(str)
posts["post_chars"] = posts["post_text"].str.len()
posts["post_words"] = posts["post_text"].str.split().str.len()

post_lookup = posts[[
    "id", "agent_id", "agent_name", "submolt", "di_post", "di_binary",
    "post_chars", "post_words", "score", "comment_count", "dump_date", "post_text"
]].rename(columns={
    "id": "post_id",
    "agent_id": "post_author_id",
    "agent_name": "post_author_name",
    "score": "post_score",
    "dump_date": "post_dump_date",
})

df = comments.merge(post_lookup, on="post_id", how="left")
df = df.dropna(subset=["di_post"]).copy()

df["log_post_chars"] = safe_log1p(df["post_chars"])
df["log_post_words"] = safe_log1p(df["post_words"])
df["log_post_comment_count"] = safe_log1p(df["comment_count"])
df["log_abs_post_score"] = safe_log1p(pd.to_numeric(df["post_score"], errors="coerce").abs())

print("Merged rows:", len(df))
print("Corrective rate:", df["is_corrective"].mean())
print("DI>0 rate:", df["di_binary"].mean())
print("Unique posts:", df["post_id"].nunique())

# Binary DI models
rows = []

model_specs = {
    "binary_DI_no_controls": ["di_binary"],
    "binary_DI_plus_post_length_chars": ["di_binary", "log_post_chars"],
    "binary_DI_plus_post_length_words": ["di_binary", "log_post_words"],
    "binary_DI_plus_length_thread_score": ["di_binary", "log_post_chars", "log_post_comment_count", "log_abs_post_score"],
}

for model_name, xs in model_specs.items():
    print("\nFitting:", model_name)
    try:
        result_rows = fit_cluster_logit(df, "is_corrective", xs, cluster_col="post_id")
        for r in result_rows:
            r["model"] = model_name
            rows.append(r)
    except Exception as e:
        rows.append({"model": model_name, "term": xs[0], "error": repr(e)})
        print("ERROR:", repr(e))

binary_results = pd.DataFrame(rows)
binary_results.to_csv(OUT / "round3_binary_di_models.csv", index=False)

# Balanced audit sample for manual checking
audit_parts = []
for label in ["corrective", "affirmation", "adversarial", "neutral"]:
    sub = df[df["response_type"] == label].copy()
    if len(sub) == 0:
        continue
    n = min(150, len(sub))
    audit_parts.append(sub.sample(n=n, random_state=42))

audit = pd.concat(audit_parts, ignore_index=True)

audit["comment_excerpt"] = (
    audit["content"].fillna("").astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.slice(0, 700)
)

audit["post_excerpt"] = (
    audit["post_text"].fillna("").astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.slice(0, 700)
)

audit["manual_label_correct"] = ""
audit["manual_correct_label"] = ""
audit["reason_or_ambiguity"] = ""

audit_cols = [
    "id", "post_id", "response_type", "manual_label_correct",
    "manual_correct_label", "reason_or_ambiguity",
    "di_post", "di_binary", "di_comment",
    "submolt", "post_chars", "post_words",
    "agent_id", "agent_name", "post_author_id", "post_author_name",
    "post_dump_date", "comment_excerpt", "post_excerpt"
]

audit_cols = [c for c in audit_cols if c in audit.columns]
audit[audit_cols].to_csv(OUT / "round3_label_audit_for_manual_review.csv", index=False)

summary = []
summary.append(f"merged_rows={len(df)}")
summary.append(f"corrective_rate={df['is_corrective'].mean()}")
summary.append(f"di_binary_rate={df['di_binary'].mean()}")
summary.append(f"unique_posts={df['post_id'].nunique()}")
summary.append("")
summary.append("Binary DI model results:")
summary.append(binary_results.to_string(index=False))
summary.append("")
summary.append("Created files:")
for p in sorted(OUT.glob("*")):
    summary.append(p.name)

summary_text = "\n".join(summary)
print("\n" + summary_text)
(OUT / "round3_summary.txt").write_text(summary_text, encoding="utf-8")

print("\nDONE")
