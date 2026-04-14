import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "outputs" / "aggregated" / "summary.json"
OUT_DIR = ROOT / "outputs" / "aggregated"
COLORS = {"tiny": "#2196F3", "100M": "#4CAF50", "300M": "#FF9800", "600M": "#E91E63"}
ORDER = ["tiny", "100M", "300M", "600M"]

def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    # Fig 1: masked_PSNR degradation curves with std shading
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in ORDER:
        if b not in summary: continue
        ratios = sorted(float(r) for r in summary[b])
        means = np.array([summary[b][str(r)]["masked_psnr"]["mean"] for r in ratios])
        stds  = np.array([summary[b][str(r)]["masked_psnr"]["std"]  for r in ratios])
        ax.plot(ratios, means, "o-", color=COLORS[b], lw=2, ms=5, label=b)
        ax.fill_between(ratios, means-stds, means+stds, color=COLORS[b], alpha=0.12)
    ax.set_xlabel("Mask Ratio", fontsize=13)
    ax.set_ylabel("masked PSNR (dB)", fontsize=13)
    ax.set_title("Multi-Tile Generalization — masked PSNR (mean ± 1 std)", fontsize=12)
    ax.set_xticks([0.1*i for i in range(1, 10)])
    ax.set_xticklabels([f"{10*i}%" for i in range(1, 10)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"fig1_masked_psnr.png", dpi=150)
    plt.close()
    print(f"[plot] Saved fig1_masked_psnr.png")

    # Fig 2: MAE and SSIM side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for b in ORDER:
        if b not in summary: continue
        ratios = sorted(float(r) for r in summary[b])
        axes[0].plot(ratios, [summary[b][str(r)]["masked_mae"]["mean"] for r in ratios],
                     "o-", color=COLORS[b], lw=2, ms=5, label=b)
        axes[1].plot(ratios, [summary[b][str(r)]["masked_ssim"]["mean"] for r in ratios],
                     "o-", color=COLORS[b], lw=2, ms=5, label=b)
    for ax, title, ylabel in zip(
        axes,
        ["masked MAE vs Mask Ratio", "masked SSIM vs Mask Ratio"],
        ["masked MAE (lower is better)", "masked SSIM (higher is better)"]
    ):
        ax.set_xlabel("Mask Ratio", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([0.1*i for i in range(1, 10)])
        ax.set_xticklabels([f"{10*i}%" for i in range(1, 10)])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR/"fig2_mae_ssim.png", dpi=150)
    plt.close()
    print(f"[plot] Saved fig2_mae_ssim.png")

if __name__ == "__main__":
    main()
