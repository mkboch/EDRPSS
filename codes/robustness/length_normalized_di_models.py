from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(".").resolve()
OUT = ROOT / "journal_upgrade_outputs_round2"
OUT.mkdir(exist_ok=True)

POSTS_FILE = ROOT / "results" / "posts_with_di.csv"
COMMENTS_FILE = ROOT / "Datasets" / "comments_labeled.csv"

print("Using posts:", POSTS_FILE)
print("Using comments:", COMMENTS_FILE)

def zscore(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return s * 0
    return (s - s.mean()) / sd

def safe_log1p(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return np.log1p(s.clip(lower=0))

def fit_cluster_logit(df, y_col, x_cols, cluster_col="post_id"):
    use = df[[y_col, cluster_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
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

# Load only needed columns
post_cols = ["id", "agent_id", "agent_name", "submolt", "title", "content", "score", "comment_count", "created_at", "dump_date", "di_post"]
comment_cols = ["id", "post_id", "agent_id", "agent_name", "content", "score", "created_at", "dump_date", "response_type", "di_comment"]

posts = pd.read_csv(POSTS_FILE, usecols=lambda c: c in post_cols, low_memory=False)
comments = pd.read_csv(COMMENTS_FILE, usecols=lambda c: c in comment_cols, low_memory=False)

posts["id"] = posts["id"].astype(str)
comments["post_id"] = comments["post_id"].astype(str)
comments["response_type"] = comments["response_type"].fillna("neutral").astype(str).str.lower()

posts["di_post"] = pd.to_numeric(posts["di_post"], errors="coerce").fillna(0).clip(0, 10)

posts["post_text"] = posts["title"].fillna("").astype(str) + "\n\n" + posts["content"].fillna("").astype(str)
posts["post_chars"] = posts["post_text"].str.len()
posts["post_words"] = posts["post_text"].str.split().str.len()

# Length-normalized DI variants
posts["di_per_100_words"] = posts["di_post"] / (posts["post_words"].clip(lower=1) / 100.0)
posts["di_per_1000_chars"] = posts["di_post"] / (posts["post_chars"].clip(lower=1) / 1000.0)

# log-scaled density, less sensitive to tiny posts
posts["di_log_density_words"] = np.log1p(posts["di_post"]) / np.log1p(posts["post_words"].clip(lower=1))
posts["di_log_density_chars"] = np.log1p(posts["di_post"]) / np.log1p(posts["post_chars"].clip(lower=1))

posts["di_binary"] = (posts["di_post"] > 0).astype(int)

# residualized DI: regress raw DI on length, use residual
X_len = sm.add_constant(pd.DataFrame({
    "log_post_chars": safe_log1p(posts["post_chars"]),
    "log_post_words": safe_log1p(posts["post_words"]),
}), has_constant="add")
ols = sm.OLS(posts["di_post"].astype(float), X_len).fit()
posts["di_resid_length"] = ols.resid

lookup_cols = [
    "id", "agent_id", "agent_name", "submolt", "di_post",
    "post_chars", "post_words", "score", "comment_count",
    "dump_date", "di_per_100_words", "di_per_1000_chars",
    "di_log_density_words", "di_log_density_chars",
    "di_binary", "di_resid_length"
]

post_lookup = posts[lookup_cols].rename(columns={
    "id": "post_id",
    "agent_id": "post_author_id",
    "agent_name": "post_author_name",
    "score": "post_score",
    "dump_date": "post_dump_date",
})

df = comments.merge(post_lookup, on="post_id", how="left")
df = df.dropna(subset=["di_post"]).copy()

df["is_corrective"] = (df["response_type"] == "corrective").astype(int)
df["log_post_chars"] = safe_log1p(df["post_chars"])
df["log_post_words"] = safe_log1p(df["post_words"])
df["log_post_comment_count"] = safe_log1p(df["comment_count"])
df["log_abs_post_score"] = safe_log1p(pd.to_numeric(df["post_score"], errors="coerce").abs())

# Standardized predictors
for col in [
    "di_post", "di_per_100_words", "di_per_1000_chars",
    "di_log_density_words", "di_log_density_chars",
    "di_resid_length"
]:
    df["z_" + col] = zscore(df[col])

print("Rows:", len(df))
print("Corrective rate:", df["is_corrective"].mean())

rows = []

model_specs = {
    "raw_DI_no_controls": ["z_di_post"],
    "raw_DI_plus_length": ["z_di_post", "log_post_chars"],
    "raw_DI_plus_length_thread_score": ["z_di_post", "log_post_chars", "log_post_comment_count", "log_abs_post_score"],

    "DI_per_100_words_no_controls": ["z_di_per_100_words"],
    "DI_per_100_words_plus_length": ["z_di_per_100_words", "log_post_chars"],
    "DI_per_100_words_plus_length_thread_score": ["z_di_per_100_words", "log_post_chars", "log_post_comment_count", "log_abs_post_score"],

    "DI_per_1000_chars_no_controls": ["z_di_per_1000_chars"],
    "DI_per_1000_chars_plus_length": ["z_di_per_1000_chars", "log_post_chars"],
    "DI_per_1000_chars_plus_length_thread_score": ["z_di_per_1000_chars", "log_post_chars", "log_post_comment_count", "log_abs_post_score"],

    "DI_log_density_words_no_controls": ["z_di_log_density_words"],
    "DI_log_density_words_plus_length": ["z_di_log_density_words", "log_post_chars"],
    "DI_log_density_words_plus_length_thread_score": ["z_di_log_density_words", "log_post_chars", "log_post_comment_count", "log_abs_post_score"],

    "DI_resid_length_no_controls": ["z_di_resid_length"],
    "DI_resid_length_plus_thread_score": ["z_di_resid_length", "log_post_comment_count", "log_abs_post_score"],

    "DI_binary_plus_length_thread_score": ["di_binary", "log_post_chars", "log_post_comment_count", "log_abs_post_score"],
}

for name, xs in model_specs.items():
    print("Fitting", name)
    try:
        out = fit_cluster_logit(df, "is_corrective", xs, cluster_col="post_id")
        for r in out:
            r["model"] = name
            rows.append(r)
    except Exception as e:
        rows.append({"model": name, "term": xs[0], "error": repr(e)})

results = pd.DataFrame(rows)
results.to_csv(OUT / "round2_length_normalized_di_models.csv", index=False)

# Correlations among DI and length
corr_cols = [
    "di_post", "post_chars", "post_words", "di_per_100_words",
    "di_per_1000_chars", "di_log_density_words", "di_resid_length"
]
corr = posts[corr_cols].corr(numeric_only=True)
corr.to_csv(OUT / "round2_di_length_correlations.csv")

# Binned summaries for raw DI and density DI
for metric in ["di_post", "di_per_100_words", "di_per_1000_chars", "di_log_density_words", "di_resid_length"]:
    temp = df[["is_corrective", metric]].dropna().copy()
    try:
        temp["bin"] = pd.qcut(temp[metric].rank(method="first"), q=10, labels=[f"Q{i}" for i in range(1, 11)])
        agg = temp.groupby("bin")["is_corrective"].agg(["sum", "count", "mean"]).reset_index()
        agg.rename(columns={"sum": "n_corrective", "count": "n_comments", "mean": "p_corrective"}, inplace=True)
        agg["metric"] = metric
        agg.to_csv(OUT / f"round2_bins_{metric}.csv", index=False)
    except Exception as e:
        print("Binning failed for", metric, e)

summary = []
summary.append(f"rows={len(df)}")
summary.append(f"corrective_rate={df['is_corrective'].mean()}")
summary.append("")
summary.append("Main terms only:")
main_terms = results[
    results["term"].isin([
        "z_di_post",
        "z_di_per_100_words",
        "z_di_per_1000_chars",
        "z_di_log_density_words",
        "z_di_resid_length",
        "di_binary"
    ])
]
summary.append(main_terms[["model", "term", "beta", "odds_ratio", "or_ci_low", "or_ci_high", "p_value"]].to_string(index=False))
summary_text = "\n".join(summary)
print(summary_text)
(OUT / "round2_summary.txt").write_text(summary_text, encoding="utf-8")

print("DONE")
