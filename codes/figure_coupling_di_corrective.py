from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS

DI_CAP = 10
N_POS_BINS = 6
RNG = 7

def compute_di(text: str, action_pats, sensitive_pats, cap: int = 10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in action_pats if re.search(pat, t))
    s = sum(1 for pat in sensitive_pats if re.search(pat, t))
    return int(min(a + s, cap))

def ensure_post_di(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    if "di" in posts.columns:
        posts["di_post"] = pd.to_numeric(posts["di"], errors="coerce").fillna(0).astype(int)
    else:
        txt = (posts["title"].fillna("").astype(str) + " " + posts["content"].fillna("").astype(str)).str.strip()
        posts["di_post"] = txt.apply(lambda t: compute_di(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, DI_CAP)).astype(int)
    posts["di_post"] = posts["di_post"].clip(lower=0, upper=DI_CAP)
    return posts

def make_di_bins(posts: pd.DataFrame, n_pos_bins: int = 6) -> tuple[pd.DataFrame, list[str]]:
    posts = posts.copy()
    posts["di_bin"] = None
    posts.loc[posts["di_post"] == 0, "di_bin"] = "DI=0"

    pos_mask = posts["di_post"] > 0
    pos = posts.loc[pos_mask].copy()

    if len(pos) == 0:
        order = ["DI=0"]
        return posts, order

    q = min(n_pos_bins, len(pos))
    # rank-based qcut -> guarantees equal-frequency bins even if DI values are discrete
    ranked = pos["di_post"].rank(method="first")
    codes = pd.qcut(ranked, q=q, labels=False, duplicates="drop") + 1
    posts.loc[pos.index, "di_bin"] = ["Q{}".format(int(c)) for c in codes]

    found_q = sorted(
        [x for x in posts["di_bin"].dropna().unique().tolist() if str(x).startswith("Q")],
        key=lambda z: int(str(z)[1:])
    )
    order = ["DI=0"] + found_q
    posts["di_bin"] = pd.Categorical(posts["di_bin"], categories=order, ordered=True)
    return posts, order

def main():
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "Datasets"
    figures_dir = project_root / "Figures"
    results_dir = project_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    posts_path = datasets_dir / "posts.csv"
    cmts_path = datasets_dir / "comments_labeled.csv"

    posts = pd.read_csv(posts_path, low_memory=False)
    cmts = pd.read_csv(cmts_path, low_memory=False)

    posts["id"] = posts["id"].astype(str)
    cmts["post_id"] = cmts["post_id"].astype(str)
    cmts["response_type"] = cmts["response_type"].fillna("neutral").astype(str).str.lower()

    posts = ensure_post_di(posts)
    posts, order = make_di_bins(posts, N_POS_BINS)

    bin_map = posts.set_index("id")["di_bin"]
    di_map = posts.set_index("id")["di_post"]

    cmts["di_bin"] = cmts["post_id"].map(bin_map)
    cmts["di_post"] = cmts["post_id"].map(di_map)
    cmts = cmts.dropna(subset=["di_bin"]).copy()
    cmts["is_corrective"] = (cmts["response_type"] == "corrective").astype(int)

    # -------- Figure 1 aggregate bins --------
    agg = (
        cmts.groupby("di_bin", observed=True)["is_corrective"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_corrective", "count": "n_comments"})
        .reset_index()
    )
    agg["p"] = agg["n_corrective"] / agg["n_comments"]

    cis = agg.apply(
        lambda r: proportion_confint(int(r["n_corrective"]), int(r["n_comments"]), method="wilson"),
        axis=1
    )
    agg["ci_low"] = [x[0] for x in cis]
    agg["ci_high"] = [x[1] for x in cis]

    # keep intended order
    agg["di_bin"] = pd.Categorical(agg["di_bin"], categories=order, ordered=True)
    agg = agg.sort_values("di_bin").reset_index(drop=True)

    out_csv = results_dir / "figure1_di_corrective_bins.csv"
    agg.to_csv(out_csv, index=False)

    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    y = agg["p"].to_numpy(float)
    yerr_low = y - agg["ci_low"].to_numpy(float)
    yerr_high = agg["ci_high"].to_numpy(float) - y

    ax.errorbar(
        x, y, yerr=[yerr_low, yerr_high],
        fmt="o-", linewidth=2.6, markersize=9, capsize=7
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in agg["di_bin"]])
    ax.set_xlabel("Directive Intensity (DI) bins (posts)")
    ax.set_ylabel("P(Corrective signaling | reply)   (Wilson 95% CI)")
    ax.set_title("Corrective signaling across raw directive-marker burden strata")

    y_min = float(np.nanmin(agg["ci_low"].to_numpy())) if len(agg) else 0.0
    y_max = float(np.nanmax(agg["ci_high"].to_numpy())) if len(agg) else 1.0
    pad = max(0.006, 0.22 * (y_max - y_min + 1e-9))
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad * 1.8))

    for i, row in agg.iterrows():
        dy = 0.004 if (i % 2 == 0) else 0.010
        ax.text(
            i,
            float(row["ci_high"]) + dy,
            f"n={int(row['n_comments'])} replies",
            ha="center",
            va="bottom",
            fontsize=10.5,
            clip_on=False,
        )

    plt.tight_layout()
    fig1_png = figures_dir / "Figure_1_DI_vs_Corrective.png"
    plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
    plt.close()

    # -------- S1 scatter (post-level) --------
    post_level = (
        cmts.groupby("post_id")["is_corrective"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "post_corrective_rate", "count": "n_replies"})
        .reset_index()
    )
    post_level["di_post"] = post_level["post_id"].map(di_map)
    post_level = post_level.dropna(subset=["di_post"]).copy()
    post_level["di_post"] = pd.to_numeric(post_level["di_post"], errors="coerce")

    out_scatter_csv = results_dir / "figureS1_post_level_scatter.csv"
    post_level.to_csv(out_scatter_csv, index=False)

    # sample for readability only if enormous
    if len(post_level) > 4500:
        post_plot = post_level.sample(4500, random_state=RNG)
    else:
        post_plot = post_level.copy()

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    jitter = np.random.default_rng(RNG).normal(0, 0.025, size=len(post_plot))
    ax.scatter(
        post_plot["di_post"].to_numpy(float) + jitter,
        post_plot["post_corrective_rate"].to_numpy(float),
        alpha=0.22,
        s=26,
    )
    ax.set_xlabel("Post Directive Intensity (DI)")
    ax.set_ylabel("Post-level corrective rate among replies")
    ax.set_title("Post-level DI vs corrective signaling (scatter; SI)")
    ax.set_xlim(-0.5, 10.5)
    ax.set_xticks(range(0, 11))
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    s1_png = figures_dir / "Figure_S1_DI_vs_Corrective_Scatter.png"
    plt.savefig(s1_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] coupling outputs:")
    print(" -", fig1_png)
    print(" -", s1_png)
    print(" -", out_csv)
    print(" -", out_scatter_csv)

if __name__ == "__main__":
    main()
