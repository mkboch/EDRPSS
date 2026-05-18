# codes/exp_event_aligned_placebo.py
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS

DI_CAP = 10
RNG_SEED = 7

def compute_di(text: str, cap: int = 10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in ACTION_PATTERNS if re.search(pat, t))
    s = sum(1 for pat in SENSITIVE_PATTERNS if re.search(pat, t))
    return int(min(a + s, cap))

def find_time_col(df: pd.DataFrame) -> str | None:
    for c in ["created_utc","created_at","created_time","timestamp","time","date"]:
        if c in df.columns:
            return c
    return None

def main():
    root = Path(__file__).resolve().parent.parent
    datasets = root / "Datasets"
    results = root / "results"
    figs = root / "Figures"
    results.mkdir(exist_ok=True, parents=True)
    figs.mkdir(exist_ok=True, parents=True)

    cmts = pd.read_csv(datasets / "comments_labeled.csv", low_memory=False)
    cmts["post_id"] = cmts["post_id"].astype(str)
    cmts["response_type"] = cmts["response_type"].fillna("neutral").astype(str).str.lower()

    # Need comment text for DI(comment); try common columns
    text_col = None
    for c in ["content","text","body","comment","message"]:
        if c in cmts.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError("comments_labeled.csv has no obvious text column among: content/text/body/comment/message")

    cmts["di_comment"] = cmts[text_col].fillna("").astype(str).apply(lambda t: compute_di(t, DI_CAP)).astype(int)

    tcol = find_time_col(cmts)
    if tcol is not None:
        # try numeric time
        cmts[tcol] = pd.to_numeric(cmts[tcol], errors="coerce")
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    # Define thread = post_id; event = first corrective reply in that post
    for pid, g in cmts.groupby("post_id", sort=False):
        if len(g) < 3:
            continue

        # order
        if tcol is not None and g[tcol].notna().any():
            g2 = g.sort_values(by=tcol, kind="mergesort")
        else:
            g2 = g.reset_index(drop=True)

        idx_corrective = np.where((g2["response_type"] == "corrective").to_numpy())[0]
        if len(idx_corrective) == 0:
            continue
        t0 = int(idx_corrective[0])

        before = g2.iloc[:t0]
        after = g2.iloc[t0+1:]
        if len(before) == 0 or len(after) == 0:
            continue

        mean_before = float(before["di_comment"].mean())
        mean_after = float(after["di_comment"].mean())
        delta_true = mean_after - mean_before

        # placebo: random NON-corrective comment index as pseudo-event
        idx_non = np.where((g2["response_type"] != "corrective").to_numpy())[0]
        if len(idx_non) < 2:
            continue
        t0p = int(rng.choice(idx_non))
        if t0p == 0 or t0p == (len(g2)-1):
            continue
        before_p = g2.iloc[:t0p]
        after_p = g2.iloc[t0p+1:]
        if len(before_p) == 0 or len(after_p) == 0:
            continue
        delta_placebo = float(after_p["di_comment"].mean() - before_p["di_comment"].mean())

        rows.append({
            "post_id": pid,
            "n_total": int(len(g2)),
            "t0_index": t0,
            "t0p_index": t0p,
            "delta_true": delta_true,
            "delta_placebo": delta_placebo,
        })

    out = pd.DataFrame(rows)
    out.to_csv(results / "event_aligned_true_vs_placebo.csv", index=False)

    # Plot distributions
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(out["delta_placebo"].dropna(), bins=60, alpha=0.6, label="placebo (random non-corrective)")
    ax.hist(out["delta_true"].dropna(), bins=60, alpha=0.6, label="true (first corrective)")
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("ΔDI_comment (after event minus before event)")
    ax.set_ylabel("count (threads/posts)")
    ax.set_title("Event-aligned ΔDI: true corrective event vs placebo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "Figure_S4_event_aligned_placebo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Summary txt
    summary = [
        "Event-aligned placebo test (no human eval)",
        "=========================================",
        f"Threads(posts) used: {len(out)}",
        "",
        f"TRUE  mean delta: {out['delta_true'].mean():.6g} | median: {out['delta_true'].median():.6g}",
        f"PLAC mean delta: {out['delta_placebo'].mean():.6g} | median: {out['delta_placebo'].median():.6g}",
        "",
        "File written:",
        f" - {results / 'event_aligned_true_vs_placebo.csv'}",
        f" - {figs / 'Figure_S4_event_aligned_placebo.png'}",
    ]
    (results / "event_aligned_placebo_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("[OK] placebo outputs:")
    print(" -", results / "event_aligned_true_vs_placebo.csv")
    print(" -", results / "event_aligned_placebo_summary.txt")
    print(" -", figs / "Figure_S4_event_aligned_placebo.png")


if __name__ == "__main__":
    main()
