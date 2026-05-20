from pathlib import Path
import sys
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(".").resolve()
DATA = ROOT / "Datasets"
RESULTS = ROOT / "results"
OUT = ROOT / "journal_upgrade_outputs"
OUT.mkdir(exist_ok=True)

print("Project root:", ROOT)
print("Output folder:", OUT)

def find_file(candidates):
    for p in candidates:
        p = ROOT / p
        if p.exists():
            return p
    raise FileNotFoundError("None found: " + str(candidates))

POSTS_FILE = find_file([
    "results/posts_with_di.csv",
    "Datasets/posts_with_di.csv",
    "Datasets/posts.csv",
])

COMMENTS_FILE = find_file([
    "Datasets/comments_labeled.csv",
    "comments_labeled.csv",
])

print("Using posts:", POSTS_FILE)
print("Using comments:", COMMENTS_FILE)

# -----------------------------
# Helpers
# -----------------------------

def zscore(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - mu) / sd

def safe_log1p(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = s.clip(lower=0)
    return np.log1p(s)

def fit_cluster_logit(df, y_col, x_cols, cluster_col="post_id"):
    use = df[[y_col, cluster_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(use) < 1000:
        return None

    y = use[y_col].astype(float)
    X = sm.add_constant(use[x_cols].astype(float), has_constant="add")

    try:
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
    except Exception as e:
        return [{"term": x_cols[0], "error": repr(e), "n_attempted": len(use)}]

def text_len_chars(s):
    return s.fillna("").astype(str).str.len()

def text_len_words(s):
    return s.fillna("").astype(str).str.split().str.len()

# -----------------------------
# Load needed columns only
# -----------------------------

post_cols = pd.read_csv(POSTS_FILE, nrows=0).columns.tolist()
comment_cols = pd.read_csv(COMMENTS_FILE, nrows=0).columns.tolist()

print("Post columns:", post_cols)
print("Comment columns:", comment_cols)

needed_post = [c for c in [
    "id", "agent_id", "agent_name", "submolt", "title", "content",
    "score", "comment_count", "created_at", "dump_date", "di_post", "di"
] if c in post_cols]

needed_comment = [c for c in [
    "id", "post_id", "agent_id", "agent_name", "parent_id", "content",
    "score", "created_at", "dump_date", "response_type", "response_type_legacy", "di_comment"
] if c in comment_cols]

print("Reading posts columns:", needed_post)
posts = pd.read_csv(POSTS_FILE, usecols=needed_post, low_memory=False)

print("Reading comments columns:", needed_comment)
comments = pd.read_csv(COMMENTS_FILE, usecols=needed_comment, low_memory=False)

# Normalize
posts["id"] = posts["id"].astype(str)
comments["post_id"] = comments["post_id"].astype(str)
comments["response_type"] = comments["response_type"].fillna("neutral").astype(str).str.lower()

if "di_post" not in posts.columns:
    if "di" in posts.columns:
        posts["di_post"] = pd.to_numeric(posts["di"], errors="coerce").fillna(0)
    else:
        raise ValueError("No di_post or di column found in posts file.")

posts["di_post"] = pd.to_numeric(posts["di_post"], errors="coerce").fillna(0).clip(0, 10)

posts["post_text"] = (
    posts.get("title", pd.Series([""] * len(posts))).fillna("").astype(str)
    + "\n\n"
    + posts.get("content", pd.Series([""] * len(posts))).fillna("").astype(str)
)

posts["post_chars"] = text_len_chars(posts["post_text"])
posts["post_words"] = text_len_words(posts["post_text"])

if "comment_count" not in posts.columns:
    posts["comment_count"] = np.nan
if "score" not in posts.columns:
    posts["score"] = np.nan
if "dump_date" not in posts.columns:
    posts["dump_date"] = np.nan
if "agent_id" not in posts.columns:
    posts["agent_id"] = np.nan
if "submolt" not in posts.columns:
    posts["submolt"] = np.nan

post_lookup = posts[[
    "id", "agent_id", "agent_name", "submolt", "di_post",
    "post_chars", "post_words", "score", "comment_count",
    "created_at", "dump_date"
]].rename(columns={
    "id": "post_id",
    "agent_id": "post_author_id",
    "agent_name": "post_author_name",
    "score": "post_score",
    "created_at": "post_created_at",
    "dump_date": "post_dump_date",
})

df = comments.merge(post_lookup, on="post_id", how="left")
df = df.dropna(subset=["di_post"]).copy()

df["is_corrective"] = (df["response_type"] == "corrective").astype(int)
df["z_di_post"] = zscore(df["di_post"])
df["log_post_chars"] = safe_log1p(df["post_chars"])
df["log_post_words"] = safe_log1p(df["post_words"])
df["log_post_comment_count"] = safe_log1p(df["comment_count"])
df["log_abs_post_score"] = safe_log1p(pd.to_numeric(df["post_score"], errors="coerce").abs())

# Use post dump_date if available, otherwise comment dump_date
if "post_dump_date" in df.columns:
    df["analysis_dump_date"] = df["post_dump_date"].fillna(df.get("dump_date", np.nan))
else:
    df["analysis_dump_date"] = df.get("dump_date", np.nan)

df["analysis_dump_date"] = df["analysis_dump_date"].astype(str)

print("Merged rows:", len(df))
print("Corrective rate:", df["is_corrective"].mean())
print("Unique posts:", df["post_id"].nunique())
print("Unique post authors:", df["post_author_id"].nunique())
print("Dump dates:", df["analysis_dump_date"].nunique())

# -----------------------------
# Experiment 1: adjusted models
# -----------------------------

adjusted_rows = []

model_specs = {
    "M0_baseline_di_only": ["z_di_post"],
    "M1_plus_post_length": ["z_di_post", "log_post_chars"],
    "M2_plus_thread_size": ["z_di_post", "log_post_chars", "log_post_comment_count"],
    "M3_plus_score": ["z_di_post", "log_post_chars", "log_post_comment_count", "log_abs_post_score"],
}

for model_name, xs in model_specs.items():
    print("Fitting", model_name, xs)
    rows = fit_cluster_logit(df, "is_corrective", xs, cluster_col="post_id")
    if rows:
        for r in rows:
            r["model"] = model_name
            adjusted_rows.append(r)

adjusted = pd.DataFrame(adjusted_rows)
adjusted.to_csv(OUT / "experiment1_adjusted_models.csv", index=False)
print("Saved experiment1_adjusted_models.csv")

# -----------------------------
# Experiment 2: temporal replication by dump_date
# -----------------------------

temporal_rows = []
date_counts = df["analysis_dump_date"].value_counts(dropna=False)

for d, n in date_counts.items():
    if str(d).lower() in ["nan", "nat", "none"]:
        continue
    if n < 5000:
        continue
    sub = df[df["analysis_dump_date"] == d].copy()
    if sub["is_corrective"].nunique() < 2 or sub["di_post"].nunique() < 2:
        continue

    sub["z_di_post_slice"] = zscore(sub["di_post"])
    print("Temporal slice:", d, "n=", len(sub))
    rows = fit_cluster_logit(sub, "is_corrective", ["z_di_post_slice"], cluster_col="post_id")
    if rows:
        r = rows[0]
        r["dump_date"] = d
        r["n_comments_slice"] = int(len(sub))
        r["n_posts_slice"] = int(sub["post_id"].nunique())
        r["corrective_rate_slice"] = float(sub["is_corrective"].mean())
        temporal_rows.append(r)

temporal = pd.DataFrame(temporal_rows)
temporal.to_csv(OUT / "experiment2_temporal_replication.csv", index=False)
print("Saved experiment2_temporal_replication.csv")

# -----------------------------
# Experiment 3: top post-author exclusion
# -----------------------------

author_rows = []
author_counts = df.groupby("post_author_id").size().sort_values(ascending=False)
author_counts = author_counts[author_counts.index.notna()]

for pct in [0, 0.01, 0.05, 0.10]:
    if pct == 0:
        sub = df.copy()
        removed_authors = 0
    else:
        n_remove = max(1, int(np.ceil(len(author_counts) * pct)))
        remove_ids = set(author_counts.head(n_remove).index.astype(str))
        sub = df[~df["post_author_id"].astype(str).isin(remove_ids)].copy()
        removed_authors = n_remove

    if len(sub) < 1000:
        continue

    sub["z_di_post_sub"] = zscore(sub["di_post"])
    print("Top-author exclusion:", pct, "n=", len(sub), "removed_authors=", removed_authors)
    rows = fit_cluster_logit(sub, "is_corrective", ["z_di_post_sub"], cluster_col="post_id")
    if rows:
        r = rows[0]
        r["exclusion"] = f"remove_top_{int(pct*100)}pct_post_authors" if pct > 0 else "none"
        r["removed_authors"] = int(removed_authors)
        r["n_comments_after_exclusion"] = int(len(sub))
        r["n_post_authors_after_exclusion"] = int(sub["post_author_id"].nunique())
        author_rows.append(r)

authors = pd.DataFrame(author_rows)
authors.to_csv(OUT / "experiment3_top_author_exclusion.csv", index=False)
print("Saved experiment3_top_author_exclusion.csv")

# -----------------------------
# Experiment 4: submolt/topic robustness
# -----------------------------

submolt_rows = []
if "submolt" in df.columns:
    sub_counts = df["submolt"].fillna("missing").astype(str).value_counts()
    for submolt, n in sub_counts.items():
        if n < 5000:
            continue
        sub = df[df["submolt"].fillna("missing").astype(str) == submolt].copy()
        if sub["is_corrective"].nunique() < 2 or sub["di_post"].nunique() < 2:
            continue
        sub["z_di_post_submolt"] = zscore(sub["di_post"])
        print("Submolt:", submolt, "n=", len(sub))
        rows = fit_cluster_logit(sub, "is_corrective", ["z_di_post_submolt"], cluster_col="post_id")
        if rows:
            r = rows[0]
            r["submolt"] = submolt
            r["n_comments_submolt"] = int(len(sub))
            r["n_posts_submolt"] = int(sub["post_id"].nunique())
            r["corrective_rate_submolt"] = float(sub["is_corrective"].mean())
            submolt_rows.append(r)

submolts = pd.DataFrame(submolt_rows)
submolts.to_csv(OUT / "experiment4_submolt_replication.csv", index=False)
print("Saved experiment4_submolt_replication.csv")

# -----------------------------
# Experiment 5: balanced label audit sample
# -----------------------------

audit_parts = []
rng = 42

for label in ["corrective", "affirmation", "adversarial", "neutral"]:
    sub = df[df["response_type"] == label].copy()
    if len(sub) == 0:
        continue
    n = min(100, len(sub))
    sample = sub.sample(n=n, random_state=rng)
    audit_parts.append(sample)

audit = pd.concat(audit_parts, ignore_index=True)

audit["comment_excerpt"] = audit["content"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 500)
audit["post_excerpt"] = (
    audit["post_id"].map(posts.set_index("id")["post_text"])
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.slice(0, 500)
)

audit_cols = [
    "id", "post_id", "response_type", "response_type_legacy",
    "di_comment", "di_post", "agent_id", "agent_name",
    "post_author_id", "post_author_name", "analysis_dump_date",
    "comment_excerpt", "post_excerpt"
]
audit_cols = [c for c in audit_cols if c in audit.columns]
audit[audit_cols].to_csv(OUT / "experiment5_balanced_label_audit_sample.csv", index=False)
print("Saved experiment5_balanced_label_audit_sample.csv")

# -----------------------------
# Summary
# -----------------------------

summary_lines = []
summary_lines.append(f"merged_rows={len(df)}")
summary_lines.append(f"unique_posts={df['post_id'].nunique()}")
summary_lines.append(f"unique_post_authors={df['post_author_id'].nunique()}")
summary_lines.append(f"corrective_rate={df['is_corrective'].mean()}")
summary_lines.append("")
summary_lines.append("response_type_counts:")
summary_lines.append(df["response_type"].value_counts(dropna=False).to_string())
summary_lines.append("")
summary_lines.append("di_post_summary:")
summary_lines.append(df["di_post"].describe().to_string())
summary_lines.append("")
summary_lines.append("outputs:")
for p in sorted(OUT.glob("*.csv")):
    summary_lines.append(str(p.name))

summary = "\n".join(summary_lines)
print(summary)
(OUT / "journal_upgrade_summary.txt").write_text(summary, encoding="utf-8")

print("\nDONE. Upload the zip file created by the next PowerShell command.")
