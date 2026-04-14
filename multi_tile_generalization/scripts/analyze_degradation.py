"""
analyze_degradation.py
----------------------
Computes two degradation thresholds from summary.json:

1. Absolute threshold — at what mask ratio does masked_PSNR drop below 30 dB
   (30 dB is the standard "acceptable reconstruction quality" threshold in image restoration)

2. Steepest drop — at what mask ratio does the per-step PSNR loss accelerate most
   (second derivative peak = inflection point = where degradation goes from gradual to steep)

Saves:
  outputs/aggregated/fig3_degradation_analysis.png
  outputs/aggregated/degradation_thresholds.json
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "outputs" / "aggregated" / "summary.json"
OUT_DIR = ROOT / "outputs" / "aggregated"

COLORS = {"tiny": "#2196F3", "100M": "#4CAF50", "300M": "#FF9800", "600M": "#E91E63"}
ORDER = ["tiny", "100M", "300M", "600M"]
ABSOLUTE_THRESHOLD_DB = 30.0


def load_summary():
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def compute_thresholds(summary):
    results = {}
    for backbone in ORDER:
        if backbone not in summary:
            continue
        ratios = sorted(float(r) for r in summary[backbone])
        means  = np.array([summary[backbone][str(r)]["masked_psnr"]["mean"] for r in ratios])

        # ── Absolute threshold ────────────────────────────────────────────────
        # First ratio where PSNR drops below 30 dB
        below = [r for r, m in zip(ratios, means) if m < ABSOLUTE_THRESHOLD_DB]
        absolute_threshold = below[0] if below else None

        # ── Steepest drop (inflection point) ─────────────────────────────────
        # Per-step drop: how many dB lost at each step
        drops = np.diff(means)           # negative values (PSNR is dropping)
        # Second derivative: where drop rate is accelerating most
        accel = np.diff(drops)           # most negative = steepest acceleration
        inflection_idx = int(np.argmin(accel)) + 1  # +1 because diff shrinks array
        inflection_ratio = ratios[inflection_idx]
        inflection_psnr  = float(means[inflection_idx])

        results[backbone] = {
            "ratios": ratios,
            "means":  means.tolist(),
            "drops":  drops.tolist(),
            "absolute_threshold_ratio": absolute_threshold,
            "absolute_threshold_psnr":  float(ABSOLUTE_THRESHOLD_DB),
            "inflection_ratio": inflection_ratio,
            "inflection_psnr":  inflection_psnr,
        }

        print(f"\n{backbone}:")
        print(f"  Absolute threshold (< {ABSOLUTE_THRESHOLD_DB} dB): "
              f"{'never' if absolute_threshold is None else f'{absolute_threshold:.0%} masking'}")
        print(f"  Steepest degradation starts at: {inflection_ratio:.0%} masking "
              f"(PSNR={inflection_psnr:.2f} dB)")
        print(f"  Per-step drops (dB): "
              f"{[round(d, 2) for d in drops.tolist()]}")

    return results


def plot_degradation_analysis(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left panel: PSNR curves with threshold annotations ───────────────────
    ax = axes[0]
    for backbone in ORDER:
        if backbone not in results: continue
        r = results[backbone]
        ratios, means = r["ratios"], r["means"]
        color = COLORS[backbone]

        ax.plot(ratios, means, "o-", color=color, lw=2, ms=5, label=backbone)

        # Mark absolute threshold point
        if r["absolute_threshold_ratio"] is not None:
            thresh_idx = ratios.index(r["absolute_threshold_ratio"])
            ax.axvline(x=r["absolute_threshold_ratio"], color=color,
                       linestyle=":", alpha=0.4, lw=1)
            ax.plot(r["absolute_threshold_ratio"], means[thresh_idx],
                    "v", color=color, ms=10, zorder=5)

        # Mark inflection point
        inf_idx = ratios.index(r["inflection_ratio"])
        ax.plot(r["inflection_ratio"], means[inf_idx],
                "*", color=color, ms=14, zorder=6)

    # 30 dB threshold line
    ax.axhline(y=ABSOLUTE_THRESHOLD_DB, color="gray", linestyle="--",
               lw=1.5, label=f"{ABSOLUTE_THRESHOLD_DB} dB threshold")

    ax.set_xlabel("Mask Ratio", fontsize=12)
    ax.set_ylabel("masked PSNR (dB)", fontsize=12)
    ax.set_title("Degradation Thresholds\n★ = steepest drop starts   ▼ = below 30 dB", fontsize=11)
    ax.set_xticks([0.1*i for i in range(1, 10)])
    ax.set_xticklabels([f"{10*i}%" for i in range(1, 10)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Right panel: per-step PSNR drop (how fast is degradation) ────────────
    ax = axes[1]
    step_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # midpoints of steps

    for backbone in ORDER:
        if backbone not in results: continue
        drops = np.abs(results[backbone]["drops"])  # positive = bigger drop
        color = COLORS[backbone]
        ax.plot(step_ratios, drops, "o-", color=color, lw=2, ms=5, label=backbone)

        # Mark steepest drop point
        inf_ratio = results[backbone]["inflection_ratio"]
        if inf_ratio in step_ratios:
            inf_drop = drops[step_ratios.index(inf_ratio)]
            ax.plot(inf_ratio, inf_drop, "*", color=color, ms=14, zorder=6)

    ax.set_xlabel("Mask Ratio", fontsize=12)
    ax.set_ylabel("PSNR drop per step (dB)", fontsize=12)
    ax.set_title("Per-Step Degradation Rate\n★ = inflection point (drop accelerates most here)", fontsize=11)
    ax.set_xticks([0.1*i for i in range(2, 10)])
    ax.set_xticklabels([f"{10*i}%" for i in range(2, 10)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Prithvi EO 2.0 — Reconstruction Quality Degradation Analysis\n(500 chips, continental US)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = OUT_DIR / "fig3_degradation_analysis.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n[plot] Saved → {out}")


def save_thresholds_json(results):
    out = {
        backbone: {
            "absolute_threshold_ratio": r["absolute_threshold_ratio"],
            "inflection_ratio": r["inflection_ratio"],
            "inflection_psnr": r["inflection_psnr"],
        }
        for backbone, r in results.items()
    }
    path = OUT_DIR / "degradation_thresholds.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] Saved → {path}")


def main():
    summary = load_summary()
    print("=" * 60)
    print("Degradation Threshold Analysis")
    print("=" * 60)
    results = compute_thresholds(summary)
    plot_degradation_analysis(results)
    save_thresholds_json(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
