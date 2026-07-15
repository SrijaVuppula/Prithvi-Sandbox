"""
build_block_aggregates.py
-------------------------
Aggregates the corrected paired block-vs-random results (results_{bb}.csv) into:

  outputs/block_summary.csv                    per backbone x ratio
  outputs/sanity_check_block_vs_random.xlsx    same + per-chip sheets

Run from repo root:
    python multi_tile_generalization/block_masking_study/scripts/build_block_aggregates.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR.parent / "outputs"
BACKBONES  = ["tiny", "100M", "300M", "600M"]
RATIOS     = [0.20, 0.40, 0.60, 0.80]


def load(bb):
    p = OUT_DIR / f"results_{bb}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} — run migration first")
    df = pd.read_csv(p)
    need = {"block_psnr", "random_psnr", "delta_rand_minus_block"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p} is not the corrected paired schema (missing {need - set(df.columns)})")
    return df


def trimmed_mean(s, frac=0.05):
    v = np.sort(s.values)
    k = int(np.floor(frac * len(v)))
    return v[k:len(v) - k].mean() if len(v) - 2 * k > 0 else v.mean()


def main():
    rows = []
    for bb in BACKBONES:
        df = load(bb)
        for r in RATIOS:
            sub = df[df["mask_ratio"] == r]
            d = sub["delta_rand_minus_block"]
            rows.append({
                "backbone": bb, "mask_ratio": r, "n_chips": sub["chip"].nunique(),
                "n_trials": len(sub),
                "block_psnr_mean":  round(sub["block_psnr"].mean(), 3),
                "random_psnr_mean": round(sub["random_psnr"].mean(), 3),
                "mean_delta":         round(d.mean(), 3),
                "median_delta":       round(d.median(), 3),
                "trimmed_mean_delta": round(trimmed_mean(d), 3),
                "pct_block_harder":  round((d > 0).mean() * 100, 1),
                "pct_random_harder": round((d < 0).mean() * 100, 1),
            })
    summary = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "block_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}\n")
    print(summary.to_string(index=False))

    xlsx = OUT_DIR / "sanity_check_block_vs_random.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        for bb in BACKBONES:
            df = load(bb)
            per_chip = (df.groupby(["chip", "mask_ratio"])
                          .agg(block_psnr=("block_psnr", "mean"),
                               random_psnr=("random_psnr", "mean"),
                               delta=("delta_rand_minus_block", "mean"))
                          .reset_index())
            per_chip["winner"] = np.where(per_chip["delta"] > 0, "block_harder", "random_harder")
            per_chip.round(3).to_excel(w, sheet_name=bb, index=False)
    print(f"\nWrote {xlsx}")


if __name__ == "__main__":
    main()
