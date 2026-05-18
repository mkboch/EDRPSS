from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS

DI_CAP = 10

def compute_di(text: str, action_pats, sensitive_pats, cap: int = 10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in action_pats if re.search(pat, t))
    s = sum(1 for pat in sensitive_pats if re.search(pat, t))
    return int(min(a + s, cap))

def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=0))
    if (not np.isfinite(sd)) or sd == 0:
        return x * 0.0
    return (x - mu) / sd

def fit_cluster_glm_binomial(y: pd.Series, x: pd.Series, clusters: pd.Series, var_name: str) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    clusters = clusters.astype(str)

    df = pd.DataFrame({
        "y": y,
        var_name: x,
        "cluster": clusters,
    }).dropna().copy()

    X = sm.add_constant(df[[var_name]], has_constant="add")
    model = sm.GLM(df["y"], X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["cluster"]})

    beta = float(res.params[var_name])
    se = float(res.bse[var_name])
    pval = float(res.pvalues[var_name])
    or_ = float(np.exp(beta))
    lo = float(np.exp(beta - 1.96 * se))
    hi = float(np.exp(beta + 1.96 * se))
    return {
        "beta": beta,
        "se_cluster_post": se,
        "p_value": pval,
        "odds_ratio": or_,
        "or_ci_low": lo,
        "or_ci_high": hi,
        "n_used": int(len(df)),
    }

def main():
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "Datasets"
    results_dir = project_root / "results"
    figures_dir = project_root / "Figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    posts = pd.read_csv(datasets_dir / "posts.csv", low_memory=False)
    cmts = pd.read_csv(datasets_dir / "comments_labeled.csv", low_memory=False)

    posts["id"] = posts["id"].astype(str)
    cmts["post_id"] = cmts["post_id"].astype(str)
    cmts["response_type"] = cmts["response_type"].fillna("neutral").astype(str).str.lower()

    # Ensure di_post exists
    if "di" in posts.columns:
        posts["di_post"] = pd.to_numeric(posts["di"], errors="coerce").fillna(0).astype(int)
    else:
        text_post = (posts["title"].fillna("").astype(str) + " " + posts["content"].fillna("").astype(str)).str.strip()
        posts["di_post"] = text_post.apply(lambda t: compute_di(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, DI_CAP)).astype(int)

    # variants at post level
    posts_tc = (posts["title"].fillna("").astype(str) + " " + posts["content"].fillna("").astype(str)).str.strip()

    def di_action_only(t: str) -> int:
        tt = (t or "").lower()
        a = sum(1 for pat in ACTION_PATTERNS if re.search(pat, tt))
        return int(min(a, DI_CAP))

    def di_sensitive_only(t: str) -> int:
        tt = (t or "").lower()
        s = sum(1 for pat in SENSITIVE_PATTERNS if re.search(pat, tt))
        return int(min(s, DI_CAP))

    posts["di_action_only"] = posts_tc.apply(di_action_only).astype(int)
    posts["di_sensitive_only"] = posts_tc.apply(di_sensitive_only).astype(int)
    posts["di_binary"] = (posts["di_post"] > 0).astype(int)

    lookup = posts.set_index("id")[["di_post", "di_action_only", "di_sensitive_only", "di_binary"]]
    cmts = cmts.merge(lookup, how="left", left_on="post_id", right_index=True)
    cmts = cmts.dropna(subset=["di_post"]).copy()

    # ========================================================
    # S2: response-type associations (reframed from "negative controls")
    # ========================================================
    x_main = zscore(cmts["di_post"])
    outcomes = [
        ("corrective", (cmts["response_type"] == "corrective").astype(int)),
        ("affirmation", (cmts["response_type"] == "affirmation").astype(int)),
        ("adversarial", (cmts["response_type"] == "adversarial").astype(int)),
        ("neutral", (cmts["response_type"] == "neutral").astype(int)),
    ]

    rows = []
    for name, y in outcomes:
        fit = fit_cluster_glm_binomial(y=y, x=x_main, clusters=cmts["post_id"], var_name="z_di_post")
        fit["outcome"] = name
        rows.append(fit)

    df_s2 = pd.DataFrame(rows)[[
        "outcome", "beta", "se_cluster_post", "p_value",
        "odds_ratio", "or_ci_low", "or_ci_high", "n_used"
    ]]

    out_s2_new = results_dir / "upgrade_response_type_associations.csv"
    out_s2_old = results_dir / "upgrade_glm_negative_controls.csv"
    df_s2.to_csv(out_s2_new, index=False)
    df_s2.to_csv(out_s2_old, index=False)

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    xs = np.arange(len(df_s2))
    y = df_s2["odds_ratio"].to_numpy(float)
    yerr = np.vstack([
        y - df_s2["or_ci_low"].to_numpy(float),
        df_s2["or_ci_high"].to_numpy(float) - y
    ])
    ax.errorbar(xs, y, yerr=yerr, fmt="o", capsize=5)
    ax.axhline(1.0, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(df_s2["outcome"].tolist())
    ax.set_ylabel("Odds ratio for z(DI) → outcome\n(cluster-robust 95% CI by post)")
    ax.set_title("Response-type associations for DI coupling (clustered GLM)")
    plt.tight_layout()

    fig_s2_new = figures_dir / "Figure_S2_response_type_associations.png"
    fig_s2_old = figures_dir / "Figure_S2_negative_control_outcomes.png"
    plt.savefig(fig_s2_new, dpi=300, bbox_inches="tight")
    plt.savefig(fig_s2_old, dpi=300, bbox_inches="tight")
    plt.close()

    # ========================================================
    # S3: alternative DI definitions, plotted as odds ratios
    # ========================================================
    variants = [
        ("z_di_post", zscore(cmts["di_post"])),
        ("z_di_action_only", zscore(cmts["di_action_only"])),
        ("z_di_sensitive_only", zscore(cmts["di_sensitive_only"])),
        ("di_binary", pd.to_numeric(cmts["di_binary"], errors="coerce")),
    ]
    y_corr = (cmts["response_type"] == "corrective").astype(int)

    vrows = []
    for vname, vx in variants:
        fit = fit_cluster_glm_binomial(y=y_corr, x=vx, clusters=cmts["post_id"], var_name=vname)
        fit["variant"] = vname
        vrows.append(fit)

    df_s3 = pd.DataFrame(vrows)[[
        "variant", "beta", "se_cluster_post", "p_value",
        "odds_ratio", "or_ci_low", "or_ci_high", "n_used"
    ]]

    out_s3 = results_dir / "upgrade_alt_di_variants.csv"
    df_s3.to_csv(out_s3, index=False)

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    xs = np.arange(len(df_s3))
    y = df_s3["odds_ratio"].to_numpy(float)
    yerr = np.vstack([
        y - df_s3["or_ci_low"].to_numpy(float),
        df_s3["or_ci_high"].to_numpy(float) - y
    ])
    ax.errorbar(xs, y, yerr=yerr, fmt="o", capsize=5)
    ax.axhline(1.0, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(df_s3["variant"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Odds ratio for predictor → corrective\n(cluster-robust 95% CI by post)")
    ax.set_title("Alternative DI definitions (clustered GLM; odds ratios)")
    plt.tight_layout()

    fig_s3 = figures_dir / "Figure_S3_alt_DI_definitions.png"
    plt.savefig(fig_s3, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] upgrade outputs:")
    print(" -", out_s2_new)
    print(" -", out_s2_old)
    print(" -", out_s3)
    print(" -", fig_s2_new)
    print(" -", fig_s2_old)
    print(" -", fig_s3)

if __name__ == "__main__":
    main()
