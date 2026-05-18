from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "mathtext.fontset": "cm",
})

def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    in_csv = results_dir / "negative_feedback_event_aligned_threads.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing: {in_csv}")

    df = pd.read_csv(in_csv)

    need = {"n_before", "n_after", "max_di_before", "delta_mean_di"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {sorted(miss)}")

    usable = df[(df["n_before"] > 0) & (df["n_after"] > 0)].copy()
    reg = usable[usable["max_di_before"] > 0].copy()
    reg = reg.dropna(subset=["delta_mean_di"]).copy()

    print(f"[INFO] usable threads: {len(usable)}")
    print(f"[INFO] regulatable threads: {len(reg)}")

    deltas = reg["delta_mean_di"].to_numpy(dtype=float)

    rng = np.random.default_rng(0)
    y = rng.normal(loc=0.0, scale=0.08, size=len(deltas))  # just jitter for separation

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = plt.gca()

    ax.scatter(deltas, y, s=18, alpha=0.55, linewidths=0)

    ax.axvline(0.0, color="black", linewidth=2)  # no change
    med = float(np.median(deltas)) if len(deltas) else np.nan
    ax.axvline(med, color="red", linewidth=5)    # median

    ax.set_yticks([])
    ax.set_ylabel("Regulatable threads (jittered for visibility)")
    ax.set_xlabel(r"$\Delta \overline{DI}_{comment}$ (after $t_0$ minus before $t_0$)")
    ax.set_title("Event-aligned change after first corrective reply")

    fig.tight_layout()
    out_png = figures_dir / "Figure_2_EventAligned_DeltaDI.png"
    out_pdf = figures_dir / "Figure_2_EventAligned_DeltaDI.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    if len(deltas):
        print("[SUMMARY] delta_mean_di (regulatable):")
        print(f"  n = {len(deltas)}")
        print(f"  mean   = {float(np.mean(deltas))}")
        print(f"  median = {float(np.median(deltas))}")

    print(f"[OK] Wrote:\n - {out_png}\n - {out_pdf}")

if __name__ == "__main__":
    main()
