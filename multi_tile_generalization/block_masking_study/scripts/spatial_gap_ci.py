"""Spatial: median PSNR/MAE tables + chip-level bootstrap 95% CI on
Delta = median(random PSNR) - median(block PSNR), pooled over trials; chips resampled."""
import numpy as np, pandas as pd
N_BOOT, SEED = 10000, 0
rng = np.random.default_rng(SEED)
out = []
for bb in ["tiny", "100M", "300M", "600M"]:
    df = pd.read_csv(f"outputs/results_{bb}_mae_replay.csv")
    print(f"\n=== {bb}  (rows={len(df)}, chips={df.chip.nunique()}) ===")
    print(f"{'ratio':>5} {'blk_med':>8} {'rnd_med':>8} {'d_med':>6} {'95% CI':>16} "
          f"{'d_mean':>7} {'blkMAE':>8} {'rndMAE':>8}")
    for r in sorted(df.mask_ratio.unique()):
        sub = df[df.mask_ratio == r]
        chips = np.sort(sub.chip.unique())
        piv = lambda col: sub.pivot(index="chip", columns="trial", values=col).reindex(chips).to_numpy()
        b, rn = piv("block_psnr_replay"), piv("random_psnr_replay")
        d_med = np.median(rn) - np.median(b)
        d_mean = rn.mean() - b.mean()
        boots = np.empty(N_BOOT)
        for k in range(N_BOOT):
            i = rng.integers(0, len(chips), len(chips))
            boots[k] = np.median(rn[i]) - np.median(b[i])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        bm, rm = sub.block_mae.median(), sub.random_mae.median()
        print(f"{r:5.2f} {np.median(b):8.2f} {np.median(rn):8.2f} {d_med:6.2f}   [{lo:5.2f}, {hi:5.2f}] "
              f"{d_mean:7.2f} {bm:8.4f} {rm:8.4f}")
        out.append([bb, r, np.median(b), np.median(rn), d_med, lo, hi, d_mean, bm, rm,
                    (sub.block_mae > sub.random_mae).mean()])
pd.DataFrame(out, columns=["backbone", "ratio", "block_psnr_median", "random_psnr_median",
    "delta_median", "ci_lo", "ci_hi", "delta_mean", "block_mae_median", "random_mae_median",
    "frac_trials_block_mae_worse"]).to_csv("outputs/spatial_gap_ci.csv", index=False)
