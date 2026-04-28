"""
aggregate_block_results.py
--------------------------
Reads per-backbone CSVs, computes mean/std summary, prints comparison
table between block masking and random masking, saves block_summary.csv

Usage
-----
  cd ~/Prithvi/Prithvi-Sandbox
  python multi_tile_generalization/block_masking_study/scripts/aggregate_block_results.py
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR  = SCRIPT_DIR.parent
OUT_DIR    = STUDY_DIR / "outputs"

RANDOM_PSNR = {
    "tiny": {0.20: 34.57, 0.40: 33.31, 0.60: 31.61, 0.80: 29.10},
    "100M": {0.20: 35.85, 0.40: 34.16, 0.60: 32.11, 0.80: 29.34},
    "300M": {0.20: 36.32, 0.40: 34.58, 0.60: 32.43, 0.80: 29.55},
    "600M": {0.20: 37.32, 0.40: 35.56, 0.60: 33.28, 0.80: 30.19},
}

BACKBONES = ["tiny", "100M", "300M", "600M"]
RATIOS    = [0.20, 0.40, 0.60, 0.80]


def load_results(backbone):
    csv_path = OUT_DIR / f"results_{backbone}.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    data = defaultdict(list)
    for row in rows:
        data[float(row["mask_ratio"])].append(row)
    result = {}
    for ratio, group in data.items():
        result[ratio] = {}
        for metric in ["block_psnr", "block_mae", "block_ssim",
                       "global_psnr", "global_mae"]:
            vals = [float(g[metric]) for g in group
                    if g.get(metric, "") not in ("", "nan")]
            result[ratio][f"{metric}_mean"] = np.mean(vals) if vals else float("nan")
            result[ratio][f"{metric}_std"]  = np.std(vals)  if vals else float("nan")
        result[ratio]["n"] = len(group)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_agg = {}
    for bb in BACKBONES:
        rows = load_results(bb)
        if not rows:
            print(f"  WARNING: No results for {bb} — skipping")
            continue
        all_agg[bb] = aggregate(rows)

    print("\n" + "="*70)
    print("BLOCK MASKING — mean block_PSNR (dB) across 500 chips x 5 trials")
    print("="*70)
    print(f"{'Ratio':<8}" + "".join(f"{bb:>10}" for bb in BACKBONES))
    print("-"*70)
    for ratio in RATIOS:
        row_str = f"{int(ratio*100):>3}%   "
        for bb in BACKBONES:
            if bb in all_agg and ratio in all_agg[bb]:
                val = all_agg[bb][ratio].get("block_psnr_mean", float("nan"))
                row_str += f"{val:>10.2f}"
            else:
                row_str += f"{'N/A':>10}"
        print(row_str)

    print("\n" + "="*70)
    print("COMPARISON: block_PSNR vs random_PSNR (delta = block - random)")
    print("Negative delta = block masking is harder (expected)")
    print("="*70)
    print(f"{'Ratio':<8}" + "".join(f"{bb:>12}" for bb in BACKBONES))
    print("-"*70)
    for ratio in RATIOS:
        row_str = f"{int(ratio*100):>3}%   "
        for bb in BACKBONES:
            if bb in all_agg and ratio in all_agg[bb]:
                block_val  = all_agg[bb][ratio].get("block_psnr_mean", float("nan"))
                random_val = RANDOM_PSNR.get(bb, {}).get(ratio, float("nan"))
                delta = block_val - random_val
                row_str += f"{delta:>+10.2f} dB"[:12]
            else:
                row_str += f"{'N/A':>12}"
        print(row_str)

    summary_path = OUT_DIR / "block_summary.csv"
    fieldnames = ["backbone", "mask_ratio", "n",
                  "block_psnr_mean", "block_psnr_std",
                  "block_mae_mean",  "block_mae_std",
                  "block_ssim_mean", "block_ssim_std",
                  "global_psnr_mean", "global_mae_mean",
                  "random_psnr_mean", "delta_psnr"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bb in BACKBONES:
            if bb not in all_agg:
                continue
            for ratio in RATIOS:
                if ratio not in all_agg[bb]:
                    continue
                agg = all_agg[bb][ratio]
                block_psnr = agg.get("block_psnr_mean", float("nan"))
                rand_psnr  = RANDOM_PSNR.get(bb, {}).get(ratio, float("nan"))
                writer.writerow({
                    "backbone":         bb,
                    "mask_ratio":       ratio,
                    "n":                agg["n"],
                    "block_psnr_mean":  round(block_psnr, 4),
                    "block_psnr_std":   round(agg.get("block_psnr_std", 0), 4),
                    "block_mae_mean":   round(agg.get("block_mae_mean", 0), 6),
                    "block_mae_std":    round(agg.get("block_mae_std",  0), 6),
                    "block_ssim_mean":  round(agg.get("block_ssim_mean", 0), 4),
                    "block_ssim_std":   round(agg.get("block_ssim_std",  0), 4),
                    "global_psnr_mean": round(agg.get("global_psnr_mean", 0), 4),
                    "global_mae_mean":  round(agg.get("global_mae_mean",  0), 6),
                    "random_psnr_mean": rand_psnr,
                    "delta_psnr":       round(block_psnr - rand_psnr, 4)
                                        if not np.isnan(block_psnr) else "nan",
                })
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
