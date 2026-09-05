"""
Per-chip backbone ranking analysis.

Tests whether the aggregate backbone ranking (mean PSNR across all 500 chips)
matches the ranking seen on individual chips. A mismatch -- e.g. 600M winning
on the mean while ranking worst on a large share of individual chips -- is the
statistically defensible form of the "600M reversal" claim.

Averages the 5 trials per (backbone, chip, mask_ratio) cell first, then ranks
the 4 backbones within each (chip, mask_ratio).

Rank 1 = highest PSNR (best).

Usage (from repo root):
    python3 analyze_backbone_ranking.py

Outputs (written to block_masking_study/outputs/):
    backbone_ranking_aggregate.csv   -- mean PSNR + aggregate rank per backbone/ratio
    backbone_ranking_per_chip.csv    -- per-chip ranks, all 500 chips x 4 ratios
    backbone_ranking_frequency.csv   -- how often each backbone takes each rank
"""

from pathlib import Path

import pandas as pd

OUTDIR = Path("multi_tile_generalization/block_masking_study/outputs")
BACKBONES = ["tiny", "100M", "300M", "600M"]
METRICS = ["block_psnr", "random_psnr"]


def load_all():
    frames = []
    for b in BACKBONES:
        f = OUTDIR / f"results_{b}.csv"
        if not f.exists():
            raise SystemExit(f"Missing {f}")
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df["backbone"] = df["backbone"].astype(str)
    return df


def main():
    df = load_all()
    print(f"Loaded {len(df)} rows, {df['chip'].nunique()} chips, "
          f"{df['backbone'].nunique()} backbones, "
          f"{sorted(df['mask_ratio'].unique())} ratios")
    print(f"Backbone labels found: {sorted(df['backbone'].unique())}")

    cell = (df.groupby(["backbone", "chip", "mask_ratio"])[METRICS]
              .mean()
              .reset_index())

    all_freq = []
    all_agg = []
    per_chip_out = None

    for metric in METRICS:
        wide = cell.pivot_table(index=["chip", "mask_ratio"],
                                columns="backbone",
                                values=metric).reset_index()

        present = [b for b in BACKBONES if b in wide.columns]

        ranks = wide[present].rank(axis=1, ascending=False, method="min")
        ranks.columns = [f"rank_{b}" for b in present]
        wide_ranked = pd.concat([wide, ranks], axis=1)
        wide_ranked["metric"] = metric

        if per_chip_out is None:
            per_chip_out = wide_ranked
        else:
            per_chip_out = pd.concat([per_chip_out, wide_ranked],
                                     ignore_index=True)

        agg = (cell.groupby(["mask_ratio", "backbone"])[metric]
                   .mean()
                   .reset_index()
                   .rename(columns={metric: "mean_psnr"}))
        agg["aggregate_rank"] = (agg.groupby("mask_ratio")["mean_psnr"]
                                    .rank(ascending=False, method="min"))
        agg["metric"] = metric
        all_agg.append(agg)

        for ratio, grp in wide_ranked.groupby("mask_ratio"):
            n = len(grp)
            for b in present:
                counts = grp[f"rank_{b}"].value_counts()
                for r in [1, 2, 3, 4]:
                    all_freq.append({
                        "metric": metric,
                        "mask_ratio": ratio,
                        "backbone": b,
                        "rank": r,
                        "n_chips": int(counts.get(r, 0)),
                        "pct_chips": round(100.0 * counts.get(r, 0) / n, 2),
                    })

    agg_df = pd.concat(all_agg, ignore_index=True)
    freq_df = pd.DataFrame(all_freq)

    agg_df.to_csv(OUTDIR / "backbone_ranking_aggregate.csv", index=False)
    per_chip_out.to_csv(OUTDIR / "backbone_ranking_per_chip.csv", index=False)
    freq_df.to_csv(OUTDIR / "backbone_ranking_frequency.csv", index=False)

    for metric in METRICS:
        print(f"\n{'='*70}\nMETRIC: {metric}\n{'='*70}")

        a = agg_df[agg_df["metric"] == metric]
        print("\nAGGREGATE mean PSNR (dB) by mask ratio:")
        piv = a.pivot(index="mask_ratio", columns="backbone",
                      values="mean_psnr")
        cols = [b for b in BACKBONES if b in piv.columns]
        print(piv[cols].round(3).to_string())

        print("\nAGGREGATE winner per ratio:")
        for ratio, g in a.groupby("mask_ratio"):
            win = g.loc[g["aggregate_rank"] == 1, "backbone"].tolist()
            print(f"  ratio {ratio}: {', '.join(win)}")

        f = freq_df[freq_df["metric"] == metric]
        print("\nPER-CHIP rank distribution (% of 500 chips):")
        for ratio, g in f.groupby("mask_ratio"):
            print(f"\n  mask_ratio = {ratio}")
            p = g.pivot(index="backbone", columns="rank", values="pct_chips")
            idx = [b for b in BACKBONES if b in p.index]
            print(p.loc[idx].round(1).to_string())

        print("\nREVERSAL CHECK -- aggregate winner vs. its per-chip record:")
        for ratio, g in a.groupby("mask_ratio"):
            win = g.loc[g["aggregate_rank"] == 1, "backbone"].iloc[0]
            fw = f[(f["mask_ratio"] == ratio) & (f["backbone"] == win)]
            best = fw.loc[fw["rank"] == 1, "pct_chips"].iloc[0]
            worst = fw.loc[fw["rank"] == 4, "pct_chips"].iloc[0]
            print(f"  ratio {ratio}: {win} wins on the mean, but ranks "
                  f"BEST on {best}% of chips and WORST on {worst}% of chips")

    print(f"\nWrote 3 CSVs to {OUTDIR}/")


if __name__ == "__main__":
    main()
