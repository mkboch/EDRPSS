# codes/polish_all_figures.py
# ------------------------------------------------------------
# Re-plot (polish) all key figures from CSVs in ./results
# Outputs (overwrites PNGs in ./Figures):
#   Figure_1_DI_vs_Corrective.png
#   Figure_S1_DI_vs_Corrective_Scatter.png
#   Figure_S2_response_type_associations.png
#   Figure_S3_alt_DI_definitions.png
#   Figure_S4_event_aligned_placebo.png
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "Figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------- global style: fix the "giant text / clipping" problem ----------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "axes.linewidth": 1.2,
})

def _save(fig, out: Path):
    # Use constrained_layout to prevent title/label clipping.
    try:
        fig.set_constrained_layout(True)
    except Exception:
        pass
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] wrote", out)

def _ensure_bins_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Accept either k/n or (n_corrective, n_comments) or p/ci columns.
    cols = set(df.columns)

    # Normalize column names we know exist in your bins CSV:
    # ['di_bin','n_corrective','n_comments','p','ci_low','ci_high']
    if "n_corrective" in cols and "n_comments" in cols:
        df = df.copy()
        df["k"] = pd.to_numeric(df["n_corrective"], errors="coerce")
        df["n"] = pd.to_numeric(df["n_comments"], errors="coerce")
        df["p_corrective"] = pd.to_numeric(df.get("p", np.nan), errors="coerce")
        df["ci_low"] = pd.to_numeric(df.get("ci_low", np.nan), errors="coerce")
        df["ci_high"] = pd.to_numeric(df.get("ci_high", np.nan), errors="coerce")
        return df

    if "k" in cols and "n" in cols:
        df = df.copy()
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        if "p_corrective" not in df.columns and "p" in df.columns:
            df["p_corrective"] = pd.to_numeric(df["p"], errors="coerce")
        return df

    raise KeyError(f"bins csv missing expected columns. Have columns: {list(df.columns)}")

def _plot_coupling_main():
    df_bins = pd.read_csv(RESULTS / "figure1_di_corrective_bins.csv")
    df = _ensure_bins_fields(df_bins)

    # x labels
    if "di_bin" in df.columns:
        labels = df["di_bin"].astype(str).tolist()
    else:
        labels = [f"bin{i}" for i in range(len(df))]

    y = df["p_corrective"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    n = df["n"].to_numpy(dtype=float)

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    ax.errorbar(
        x, y,
        yerr=np.vstack([y - lo, hi - y]),
        fmt="o-", capsize=6, linewidth=3, markersize=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Directive intensity (DI) bins (posts)")
    ax.set_ylabel("P(corrective | reply)\n(Wilson 95% CI)")
    ax.set_title("Corrective signaling scales with directive intensity (DI)")

    # y-lims tight around CI range (prevents huge whitespace)
    y_min = float(np.nanmin(lo))
    y_max = float(np.nanmax(hi))
    pad = max(0.01, 0.15 * (y_max - y_min))
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    # cleaner n-labels (smaller, staggered, and never clipped)
    for i in range(len(x)):
        dy = 0.008 if (i % 2 == 0) else 0.016
        ax.text(
            x[i],
            float(hi[i]) + dy,
            f"n={int(n[i])}",
            ha="center",
            va="bottom",
            fontsize=12,
            clip_on=False
        )

    out = FIGS / "Figure_1_DI_vs_Corrective.png"
    _save(fig, out)

def _plot_scatter():
    # This CSV is produced by figure_coupling_di_corrective.py (you have it)
    p = RESULTS / "figureS1_post_level_scatter.csv"
    if not p.exists():
        print("[WARN] missing", p, "Skipping scatter.")
        return
    df = pd.read_csv(p)

    # Be tolerant about column names
    # expected: di_post, post_corrective_rate
    di_col = None
    y_col = None
    for c in df.columns:
        cl = c.lower()
        if di_col is None and ("di" in cl and "post" in cl):
            di_col = c
        if y_col is None and ("corrective" in cl and "rate" in cl):
            y_col = c
    if di_col is None or y_col is None:
        print("[WARN] scatter csv columns not recognized:", list(df.columns))
        return

    x = pd.to_numeric(df[di_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].to_numpy()
    y = y[m].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)

    # Smaller markers + higher alpha control (prevents the "solid blobs")
    ax.scatter(x, y, s=18, alpha=0.18)

    ax.set_xlabel("Post directive intensity (DI)")
    ax.set_ylabel("Post-level corrective rate among replies")
    ax.set_title("Post-level DI vs corrective signaling (scatter; SI)")

    ax.set_ylim(-0.02, 1.02)
    # keep x-range tight-ish but not misleading
    xlo = np.nanpercentile(x, 0.5)
    xhi = np.nanpercentile(x, 99.5)
    ax.set_xlim(min(0, xlo), xhi)

    out = FIGS / "Figure_S1_DI_vs_Corrective_Scatter.png"
    _save(fig, out)

def _plot_glm_odds_ratios(csv_path: Path, out_path: Path, title: str, xcol: str):
    df = pd.read_csv(csv_path)
    if xcol not in df.columns:
        raise KeyError(f"{csv_path.name} missing {xcol}. Have {list(df.columns)}")

    # Accept either (beta,se) or (beta_z_di,se_cluster_post) etc.
    beta_col = None
    se_col = None
    for c in df.columns:
        if c in ("beta", "beta_z_di"):
            beta_col = c
        if c in ("se", "se_cluster_post"):
            se_col = c
    if beta_col is None or se_col is None:
        raise KeyError(f"{csv_path.name} missing beta/se columns. Have {list(df.columns)}")

    xlabels = df[xcol].astype(str).tolist()
    beta = pd.to_numeric(df[beta_col], errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(df[se_col], errors="coerce").to_numpy(dtype=float)

    # OR and 95% CI on OR scale (approx)
    or_mid = np.exp(beta)
    or_lo = np.exp(beta - 1.96 * se)
    or_hi = np.exp(beta + 1.96 * se)

    x = np.arange(len(df))

    # Wider figure to prevent title clipping
    fig, ax = plt.subplots(figsize=(12.0, 5.0), constrained_layout=True)
    ax.errorbar(
        x,
        or_mid,
        yerr=np.vstack([or_mid - or_lo, or_hi - or_mid]),
        fmt="o",
        capsize=6,
        markersize=11,
        linewidth=2
    )
    ax.axhline(1.0, linewidth=1.2)
    ax.set_xticks(x)

    # Moderate rotation only; keep readable
    ax.set_xticklabels(xlabels, rotation=18, ha="right")
    ax.set_ylabel("Odds ratio (predictor → outcome)\n(cluster-robust 95% CI by post)")
    ax.set_title(title)

    # Tight y-lims around the actual CI range (fixes huge empty top space)
    ymin = float(np.nanmin(or_lo))
    ymax = float(np.nanmax(or_hi))
    pad = max(0.02, 0.15 * (ymax - ymin))
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    _save(fig, out_path)

def _plot_placebo():
    p = RESULTS / "event_aligned_true_vs_placebo.csv"
    if not p.exists():
        print("[WARN] missing", p, "Skipping placebo plot.")
        return
    df = pd.read_csv(p)

    # Try a few common column names
    cand_true = [c for c in df.columns if "true" in c.lower() and "delta" in c.lower()]
    cand_pl = [c for c in df.columns if ("placebo" in c.lower() or "random" in c.lower()) and "delta" in c.lower()]

    # If not found, fallback to first two numeric columns
    if cand_true and cand_pl:
        col_true = cand_true[0]
        col_pl = cand_pl[0]
    else:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(nums) < 2:
            # coercion fallback
            nums = []
            for c in df.columns:
                try:
                    pd.to_numeric(df[c].head(50), errors="raise")
                    nums.append(c)
                except Exception:
                    pass
        if len(nums) < 2:
            print("[WARN] could not detect placebo columns in:", list(df.columns))
            return
        col_true, col_pl = nums[0], nums[1]

    v_true = pd.to_numeric(df[col_true], errors="coerce").dropna().to_numpy(dtype=float)
    v_pl = pd.to_numeric(df[col_pl], errors="coerce").dropna().to_numpy(dtype=float)

    # Trim extreme tails for a cleaner plot (still honest): use 0.5%–99.5% over combined
    vv = np.concatenate([v_true, v_pl]) if (len(v_true) and len(v_pl)) else (v_true if len(v_true) else v_pl)
    lo = np.nanpercentile(vv, 0.5)
    hi = np.nanpercentile(vv, 99.5)

    fig, ax = plt.subplots(figsize=(12.0, 5.2), constrained_layout=True)

    bins = 40
    ax.hist(v_pl, bins=bins, range=(lo, hi), alpha=0.55, label="placebo (random non-corrective)")
    ax.hist(v_true, bins=bins, range=(lo, hi), alpha=0.55, label="true (first corrective)")

    # Means as vertical lines
    m_pl = float(np.nanmean(v_pl)) if len(v_pl) else 0.0
    m_true = float(np.nanmean(v_true)) if len(v_true) else 0.0
    ax.axvline(m_pl, linewidth=2.5)
    ax.axvline(m_true, linewidth=2.5)

    ax.set_title("Event-aligned ΔDI: true corrective event vs placebo")
    ax.set_xlabel("ΔDI_comment (after event − before event)")
    ax.set_ylabel("count (threads/posts)")
    ax.legend(loc="upper left", framealpha=0.9)

    out = FIGS / "Figure_S4_event_aligned_placebo.png"
    _save(fig, out)

def main():
    print("[INFO] polishing figures from CSVs in:", RESULTS)

    _plot_coupling_main()
    _plot_scatter()

    _plot_glm_odds_ratios(
        RESULTS / "upgrade_glm_negative_controls.csv",
        FIGS / "Figure_S2_response_type_associations.png",
        "Response-type associations for DI coupling (clustered GLM; odds ratios)",
        xcol="outcome",
    )
    _plot_glm_odds_ratios(
        RESULTS / "upgrade_alt_di_variants.csv",
        FIGS / "Figure_S3_alt_DI_definitions.png",
        "Alternative DI definitions (clustered GLM; odds ratios)",
        xcol="variant",
    )

    _plot_placebo()

    print("[DONE] Polished figures written to:", FIGS)

if __name__ == "__main__":
    main()
