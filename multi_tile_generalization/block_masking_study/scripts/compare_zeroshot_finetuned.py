"""
Compare zero-shot vs. fine-tuned PSNR, paired by (chip, mask_ratio, trial_seed).

Reads outputs/results_{backbone}.csv (zero-shot) and
outputs_finetuned/results_{backbone}.csv (fine-tuned), matches rows on the
identifying keys, and reports mean/median PSNR + the paired delta for both
block and random masking, per backbone x ratio.

Usage:
    python compare_zeroshot_finetuned.py
"""
import csv
import statistics
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
BACKBONES = ["tiny", "100M", "300M", "600M"]

def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["chip"], r["mask_ratio"], r["trial_seed"])
            rows[key] = r
    return rows

def main():
    out_rows = []
    for bb in BACKBONES:
        zs_path = STUDY_DIR / "outputs" / f"results_{bb}.csv"
        ft_path = STUDY_DIR / "outputs_finetuned" / f"results_{bb}.csv"
        zs = load(zs_path)
        ft = load(ft_path)

        matched = set(zs) & set(ft)
        missing = (set(zs) | set(ft)) - matched
        print(f"[{bb}] zero-shot rows={len(zs)} finetuned rows={len(ft)} "
              f"matched={len(matched)} unmatched={len(missing)}")

        by_ratio = {}
        for key in matched:
            ratio = key[1]
            by_ratio.setdefault(ratio, []).append((zs[key], ft[key]))

        for ratio in sorted(by_ratio, key=float):
            pairs = by_ratio[ratio]
            for geom_col in ("block_psnr", "random_psnr"):
                zs_vals = [float(z[geom_col]) for z, f in pairs]
                ft_vals = [float(f[geom_col]) for z, f in pairs]
                deltas = [f - z for z, f in zip(zs_vals, ft_vals)]
                out_rows.append({
                    "backbone": bb,
                    "mask_ratio": ratio,
                    "geometry": geom_col.replace("_psnr", ""),
                    "n": len(pairs),
                    "zeroshot_mean_psnr": round(statistics.mean(zs_vals), 3),
                    "finetuned_mean_psnr": round(statistics.mean(ft_vals), 3),
                    "delta_mean_psnr": round(statistics.mean(deltas), 3),
                    "zeroshot_median_psnr": round(statistics.median(zs_vals), 3),
                    "finetuned_median_psnr": round(statistics.median(ft_vals), 3),
                })

    out_path = STUDY_DIR / "outputs_finetuned" / "zeroshot_vs_finetuned_summary.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved: {out_path}")

    print(f"\n{'backbone':<8} {'ratio':<6} {'geom':<8} {'n':<6} "
          f"{'zs_mean':<9} {'ft_mean':<9} {'delta':<8}")
    for r in out_rows:
        print(f"{r['backbone']:<8} {r['mask_ratio']:<6} {r['geometry']:<8} "
              f"{r['n']:<6} {r['zeroshot_mean_psnr']:<9} "
              f"{r['finetuned_mean_psnr']:<9} {r['delta_mean_psnr']:<8}")

if __name__ == "__main__":
    main()
