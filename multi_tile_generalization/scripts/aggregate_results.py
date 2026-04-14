import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PER_TILE_DIR = ROOT / "outputs" / "per_tile"
AGG_DIR = ROOT / "outputs" / "aggregated"
AGG_DIR.mkdir(parents=True, exist_ok=True)
METRICS = ["mae", "psnr", "ssim", "masked_mae", "masked_psnr", "masked_ssim"]

def main():
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for csv_path in sorted(PER_TILE_DIR.glob("*_results.csv")):
        print(f"[aggregate] Reading {csv_path.name}...")
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                for m in METRICS:
                    data[row["backbone"]][float(row["mask_ratio"])][m].append(float(row[m]))

    summary = {}
    for backbone, ratios in data.items():
        summary[backbone] = {}
        for ratio in sorted(ratios):
            summary[backbone][ratio] = {}
            for m in METRICS:
                vals = ratios[ratio][m]
                summary[backbone][ratio][m] = {
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals)), 6),
                    "n": len(vals)
                }

    fieldnames = ["backbone","mask_ratio","n_chips"] + [f"{m}_{s}" for m in METRICS for s in ["mean","std"]]
    with open(AGG_DIR/"summary.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for backbone, ratios in summary.items():
            for ratio, metrics in ratios.items():
                row = {"backbone": backbone, "mask_ratio": ratio, "n_chips": metrics[METRICS[0]]["n"]}
                for m in METRICS:
                    row[f"{m}_mean"] = metrics[m]["mean"]
                    row[f"{m}_std"] = metrics[m]["std"]
                w.writerow(row)

    with open(AGG_DIR/"summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("masked_PSNR mean (dB)")
    print("="*60)
    backbones = sorted(summary.keys())
    ratios = sorted(next(iter(summary.values())).keys())
    print(f"{'Ratio':<10}" + "".join(f"{b:>10}" for b in backbones))
    print("-"*60)
    for r in ratios:
        row = f"{r:<10.0%}"
        for b in backbones:
            val = summary[b][r]["masked_psnr"]["mean"]
            row += f"{val:>10.2f}"
        print(row)
    print(f"\nSaved: {AGG_DIR}/summary.csv and summary.json")

if __name__ == "__main__":
    main()
