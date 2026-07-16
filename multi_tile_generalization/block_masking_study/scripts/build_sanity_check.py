"""
build_sanity_check.py
----------------------------
Builds the sanity check Excel workbook from the paired block-vs-random
results (results_fixed_{backbone}.csv), which contain matched
block_psnr / random_psnr / delta_rand_minus_block per chip x ratio x trial
(5 trials per chip x ratio, averaged here).

Sign convention:
    delta_random_minus_block = random_psnr - block_psnr
    delta > 0  ->  random_psnr is higher  ->  block did WORSE  ->  block_harder
    delta < 0  ->  block_psnr is higher   ->  random did WORSE ->  random_harder

Run from repo root:
    python multi_tile_generalization/block_masking_study/scripts/build_sanity_check.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
IN_DIR    = REPO_ROOT / "multi_tile_generalization" / "block_masking_study" / "outputs"
OUT_DIR   = IN_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = ["tiny", "100M", "300M", "600M"]


def trimmed_mean(x, pct=0.05):
    x = np.sort(x.values)
    k = int(len(x) * pct)
    return x[k: len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()


def main():
    per_bb = {}
    all_frames = []

    for bb in BACKBONES:
        csv_path = IN_DIR / f"results_fixed_{bb}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path} -- run run_paired_block_random.py first")

        df = pd.read_csv(csv_path)

        n_chips = df["chip"].nunique()
        if n_chips != 500:
            print(f"  WARNING {bb}: expected 500 chips, found {n_chips}")

        agg = (df.groupby(["chip", "mask_ratio"], as_index=False)
                 .agg(block_psnr=("block_psnr", "mean"),
                      random_psnr=("random_psnr", "mean"),
                      n_trials=("trial", "count")))
        agg["delta_random_minus_block"] = agg["random_psnr"] - agg["block_psnr"]
        agg["winner"] = np.where(agg["delta_random_minus_block"] > 0,
                                  "block_harder", "random_harder")
        agg["backbone"] = bb

        per_bb[bb] = agg
        all_frames.append(agg)

    all_df = pd.concat(all_frames, ignore_index=True)

    summary_rows = []
    for bb in BACKBONES:
        sub = per_bb[bb]
        for ratio in sorted(sub["mask_ratio"].unique()):
            d = sub.loc[sub["mask_ratio"] == ratio, "delta_random_minus_block"]
            summary_rows.append({
                "backbone": bb,
                "mask_ratio": ratio,
                "mean_delta": round(d.mean(), 3),
                "median_delta": round(d.median(), 3),
                "trimmed_mean_delta": round(trimmed_mean(d), 3),
                "pct_block_harder": round((d > 0).mean() * 100, 1),
                "pct_random_harder": round((d <= 0).mean() * 100, 1),
                "n_chips": len(d),
            })
    summary_df = pd.DataFrame(summary_rows)

    out_path = OUT_DIR / "sanity_check_block_vs_random.xlsx"
    cols = ["chip", "mask_ratio", "block_psnr", "random_psnr",
            "delta_random_minus_block", "winner"]
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for bb in BACKBONES:
            per_bb[bb][cols].to_excel(writer, sheet_name=bb, index=False)
        all_df[["backbone"] + cols].to_excel(writer, sheet_name="All", index=False)

    print(f"Saved: {out_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
