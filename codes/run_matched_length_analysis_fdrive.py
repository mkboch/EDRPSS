from pathlib import Path
import pandas as pd
import numpy as np
import re
import math
from statistics import NormalDist

ROOT = Path(".").resolve()

COMMENTS = Path(r"F:\Research Files\OpenClaw_V3\working_outputs\comments_labeled.csv")
POSTS_DI = Path(r"F:\Research Files\OpenClaw_V3\working_outputs\posts_with_di.csv")

OUTDIR = ROOT / "results" / "matched_length_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_POST_LEVEL = OUTDIR / "post_level_matched_length_dataset.csv"
OUT_PAIRS = OUTDIR / "matched_length_pairs.csv"
OUT_SUMMARY = OUTDIR / "matched_length_summary.csv"
OUT_REPORT = OUTDIR / "matched_length_report.txt"
OUT_TEX = ROOT / "latex_tables" / "table_matched_length.tex"
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

BOOT_N = 2000
RANDOM_SEED = 20260525
CALIPER_LOG_WORDS = math.log(1.25)  # matched posts must be within about 25% length
rng = np.random.default_rng(RANDOM_SEED)

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def word_count_text(title, content):
    txt = safe_text(title) + " " + safe_text(content)
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return 0
    return len(txt.split())

def normal_2sided_from_z(z):
    nd = NormalDist()
    return 2 * (1 - nd.cdf(abs(z)))

def sign_test_normal(pos, neg):
    n = pos + neg
    if n == 0:
        return np.nan, np.nan
    # Under H0, positive signs ~ Binomial(n, 0.5); normal approx with continuity correction
    expected = n / 2
    observed = pos
    cc = 0.5 if observed > expected else -0.5
    z = (observed - expected - cc) / math.sqrt(n * 0.25)
    p = normal_2sided_from_z(z)
    return z, p

def bootstrap_ci(x, n_boot=BOOT_N):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    means = np.empty(n_boot, dtype=float)
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = np.mean(x[idx])
    return np.quantile(means, 0.025), np.quantile(means, 0.975)

def nearest_control_indices(treated_log, control_log):
    # Matching with replacement within the same length stratum.
    # For each treated point, choose the nearest control in log-word length.
    pos = np.searchsorted(control_log, treated_log)
    left = np.clip(pos - 1, 0, len(control_log) - 1)
    right = np.clip(pos, 0, len(control_log) - 1)

    left_dist = np.abs(treated_log - control_log[left])
    right_dist = np.abs(treated_log - control_log[right])
    choose_right = right_dist < left_dist

    idx = np.where(choose_right, right, left)
    dist = np.where(choose_right, right_dist, left_dist)
    return idx, dist

def make_matches(df, treated_mask, control_mask, comparison_name):
    treated = df[treated_mask].copy()
    control = df[control_mask].copy()

    pairs = []

    for decile in sorted(df["length_decile"].dropna().unique()):
        t = treated[treated["length_decile"] == decile].copy()
        c = control[control["length_decile"] == decile].copy()

        if len(t) == 0 or len(c) == 0:
            continue

        c = c.sort_values("log_words").reset_index(drop=True)
        t = t.sort_values("log_words").reset_index(drop=True)

        c_log = c["log_words"].to_numpy(float)
        t_log = t["log_words"].to_numpy(float)

        idx, dist = nearest_control_indices(t_log, c_log)

        keep = dist <= CALIPER_LOG_WORDS
        if keep.sum() == 0:
            continue

        t_keep = t.loc[keep].reset_index(drop=True)
        c_keep = c.iloc[idx[keep]].reset_index(drop=True)
        dist_keep = dist[keep]

        tmp = pd.DataFrame({
            "comparison": comparison_name,
            "length_decile": decile,
            "treated_post_id": t_keep["post_id"].values,
            "control_post_id": c_keep["post_id"].values,
            "treated_di": t_keep["di_post"].values,
            "control_di": c_keep["di_post"].values,
            "treated_words": t_keep["word_count"].values,
            "control_words": c_keep["word_count"].values,
            "abs_log_word_diff": dist_keep,
            "treated_replies": t_keep["n_replies"].values,
            "control_replies": c_keep["n_replies"].values,
            "treated_corrective": t_keep["n_corrective"].values,
            "control_corrective": c_keep["n_corrective"].values,
            "treated_rate": t_keep["corrective_rate"].values,
            "control_rate": c_keep["corrective_rate"].values,
        })
        tmp["rate_diff"] = tmp["treated_rate"] - tmp["control_rate"]
        tmp["word_diff"] = tmp["treated_words"] - tmp["control_words"]
        pairs.append(tmp)

    if not pairs:
        return pd.DataFrame()

    return pd.concat(pairs, ignore_index=True)

def summarize_pairs(pairs, comparison_name):
    if len(pairs) == 0:
        return {
            "comparison": comparison_name,
            "matched_pairs": 0,
            "mean_rate_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "median_rate_diff": np.nan,
            "p_positive": np.nan,
            "p_negative": np.nan,
            "sign_z": np.nan,
            "sign_p": np.nan,
            "mean_abs_log_word_diff": np.nan,
            "mean_abs_word_diff": np.nan,
            "treated_mean_rate": np.nan,
            "control_mean_rate": np.nan,
        }

    diff = pairs["rate_diff"].to_numpy(float)
    ci_low, ci_high = bootstrap_ci(diff)

    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    zero = int((diff == 0).sum())
    z, p = sign_test_normal(pos, neg)

    return {
        "comparison": comparison_name,
        "matched_pairs": len(pairs),
        "mean_rate_diff": float(np.mean(diff)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "median_rate_diff": float(np.median(diff)),
        "p_positive": float(pos / len(diff)),
        "p_negative": float(neg / len(diff)),
        "p_zero": float(zero / len(diff)),
        "sign_z": float(z) if not pd.isna(z) else np.nan,
        "sign_p": float(p) if not pd.isna(p) else np.nan,
        "mean_abs_log_word_diff": float(pairs["abs_log_word_diff"].mean()),
        "mean_abs_word_diff": float(np.abs(pairs["word_diff"]).mean()),
        "treated_mean_rate": float(pairs["treated_rate"].mean()),
        "control_mean_rate": float(pairs["control_rate"].mean()),
        "treated_mean_words": float(pairs["treated_words"].mean()),
        "control_mean_words": float(pairs["control_words"].mean()),
    }

print("=" * 100)
print("MATCHED-LENGTH ANALYSIS")
print("=" * 100)
print("ROOT:", ROOT)
print("COMMENTS:", COMMENTS)
print("POSTS_DI:", POSTS_DI)

if not COMMENTS.exists():
    raise FileNotFoundError(COMMENTS)
if not POSTS_DI.exists():
    raise FileNotFoundError(POSTS_DI)

# ------------------------------------------------------------
# 1. Aggregate comments to post level
# ------------------------------------------------------------
print("\n[1] Aggregating comments to post level...")

agg_parts = []
for k, chunk in enumerate(pd.read_csv(COMMENTS, usecols=["post_id", "response_type"], chunksize=500000, low_memory=False), start=1):
    chunk["response_type"] = chunk["response_type"].fillna("neutral").astype(str).str.lower()
    chunk["is_corrective"] = (chunk["response_type"] == "corrective").astype(int)
    g = chunk.groupby("post_id", dropna=False)["is_corrective"].agg(["sum", "count"]).reset_index()
    g = g.rename(columns={"sum": "n_corrective", "count": "n_replies"})
    agg_parts.append(g)
    print(f"  comments chunk {k}: rows={len(chunk):,}")

comment_agg = pd.concat(agg_parts, ignore_index=True)
comment_agg = comment_agg.groupby("post_id", dropna=False).agg({
    "n_corrective": "sum",
    "n_replies": "sum",
}).reset_index()

comment_agg["corrective_rate"] = comment_agg["n_corrective"] / comment_agg["n_replies"]
comment_agg["post_id"] = comment_agg["post_id"].astype(str)

post_ids = set(comment_agg["post_id"])
print("  post-level rows with replies:", len(comment_agg))

# ------------------------------------------------------------
# 2. Read post features only for posts with replies
# ------------------------------------------------------------
print("\n[2] Building post-level feature table from posts_with_di.csv...")

feature_parts = []
usecols = ["id", "submolt", "title", "content", "score", "comment_count", "di_post"]

for k, chunk in enumerate(pd.read_csv(POSTS_DI, usecols=usecols, chunksize=100000, low_memory=False), start=1):
    chunk["id"] = chunk["id"].astype(str)
    sub = chunk[chunk["id"].isin(post_ids)].copy()

    if len(sub):
        sub["word_count"] = [
            word_count_text(t, c)
            for t, c in zip(sub["title"], sub["content"])
        ]
        sub["char_count"] = (sub["title"].fillna("").astype(str) + " " + sub["content"].fillna("").astype(str)).str.len()
        sub = sub.rename(columns={"id": "post_id"})
        sub["di_post"] = pd.to_numeric(sub["di_post"], errors="coerce").fillna(0).astype(int)
        sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
        sub["comment_count_archive"] = pd.to_numeric(sub["comment_count"], errors="coerce")
        feature_parts.append(sub[[
            "post_id", "submolt", "di_post", "word_count", "char_count", "score", "comment_count_archive"
        ]])

    if k % 10 == 0:
        print(f"  posts chunk {k}: matched feature rows so far={sum(len(x) for x in feature_parts):,}")

features = pd.concat(feature_parts, ignore_index=True)
print("  feature rows:", len(features))

# ------------------------------------------------------------
# 3. Merge and define length strata
# ------------------------------------------------------------
print("\n[3] Merging post features with comment aggregation...")

df = comment_agg.merge(features, on="post_id", how="inner")
df = df[(df["word_count"] > 0) & df["di_post"].notna()].copy()
df["log_words"] = np.log1p(df["word_count"])
df["has_di"] = (df["di_post"] > 0).astype(int)

# qcut may drop duplicate boundaries if many same lengths
df["length_decile"] = pd.qcut(df["log_words"], q=10, labels=False, duplicates="drop")

df.to_csv(OUT_POST_LEVEL, index=False)

print("  merged rows:", len(df))
print("  total replies represented:", int(df["n_replies"].sum()))
print("  total corrective represented:", int(df["n_corrective"].sum()))
print("  post clusters:", df["post_id"].nunique())
print("  saved post-level dataset:", OUT_POST_LEVEL)

# ------------------------------------------------------------
# 4. Run matched comparisons
# ------------------------------------------------------------
print("\n[4] Running matched comparisons...")

comparisons = [
    ("DI>0 vs DI=0", df["di_post"] > 0, df["di_post"] == 0),
    ("DI>=2 vs DI=0", df["di_post"] >= 2, df["di_post"] == 0),
    ("DI>=3 vs DI=0", df["di_post"] >= 3, df["di_post"] == 0),
]

all_pairs = []
summaries = []

for name, treated_mask, control_mask in comparisons:
    print("  comparison:", name)
    pairs = make_matches(df, treated_mask, control_mask, name)
    print("    matched pairs:", len(pairs))
    all_pairs.append(pairs)
    summaries.append(summarize_pairs(pairs, name))

pairs_all = pd.concat(all_pairs, ignore_index=True)
summary = pd.DataFrame(summaries)

pairs_all.to_csv(OUT_PAIRS, index=False)
summary.to_csv(OUT_SUMMARY, index=False)

# ------------------------------------------------------------
# 5. Write LaTeX table
# ------------------------------------------------------------
def fmt3(x):
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"

def fmt_p(x):
    if pd.isna(x):
        return "--"
    if x < 0.001:
        return "$<0.001$"
    return f"{x:.3f}"

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{Matched-length comparison of corrective-reply rates.} Posts with higher DI are matched to posts with DI=0 within post-length strata using nearest-neighbor matching on log word count. Differences are treated minus matched control corrective-reply rate.}
\label{tab:matched_length}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Comparison & Pairs & Treated rate & Control rate & Mean diff. [95\% CI] & Sign-test $p$ \\
\midrule
""")
    for _, r in summary.iterrows():
        ci = f"{fmt3(r['mean_rate_diff'])} [{fmt3(r['ci_low'])}, {fmt3(r['ci_high'])}]"
        f.write(
            f"{r['comparison']} & {int(r['matched_pairs']):,} & "
            f"{fmt3(r['treated_mean_rate'])} & {fmt3(r['control_mean_rate'])} & "
            f"{ci} & {fmt_p(r['sign_p'])} \\\\\n"
        )
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------
# 6. Write text report
# ------------------------------------------------------------
lines = []
lines.append("=" * 100)
lines.append("MATCHED-LENGTH ANALYSIS REPORT")
lines.append("=" * 100)
lines.append(f"Root: {ROOT}")
lines.append(f"Input comments: {COMMENTS}")
lines.append(f"Input posts_with_di: {POSTS_DI}")
lines.append(f"Post-level eligible rows: {len(df):,}")
lines.append(f"Total replies represented: {int(df['n_replies'].sum()):,}")
lines.append(f"Total corrective represented: {int(df['n_corrective'].sum()):,}")
lines.append(f"Caliper log words: {CALIPER_LOG_WORDS:.6f} (~within 25% word length)")
lines.append("")
lines.append("Summary:")
lines.append(summary.to_string(index=False))
lines.append("")
lines.append("Outputs:")
lines.append(str(OUT_POST_LEVEL))
lines.append(str(OUT_PAIRS))
lines.append(str(OUT_SUMMARY))
lines.append(str(OUT_TEX))

report = "\n".join(lines)
OUT_REPORT.write_text(report, encoding="utf-8")

print("\nDONE")
print(report)
