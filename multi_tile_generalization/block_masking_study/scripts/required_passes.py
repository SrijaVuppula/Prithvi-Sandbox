"""Required number of passes per condition/metric via n = (z*sigma/E)^2. Saves CSV."""
import pandas as pd, numpy as np, math, csv

CSV = "multi_tile_generalization/block_masking_study/outputs/inference_energy.csv"
OUT = "multi_tile_generalization/block_masking_study/outputs/required_passes.csv"
Z = 1.96
E = {"time": 0.5, "enc": 0.5, "dec": 0.5, "power": 5.0, "energy": 50.0}

df = pd.read_csv(CSV)
rows = []
print(f"{'backbone':<8}{'ratio':>6}{'type':>8}", end="")
for m in E: print(f"{m+'_n':>10}", end="")
print()
for _, r in df.iterrows():
    row = {"backbone": r["backbone"], "mask_ratio": r["mask_ratio"], "mask_type": r["mask_type"]}
    print(f"{r['backbone']:<8}{int(r['mask_ratio']*100):>5}%{r['mask_type']:>8}", end="")
    for m, e in E.items():
        sd = r[f"{m}_std"]
        n = math.ceil((Z*sd/e)**2) if sd > 0 else 1
        row[f"{m}_required_n"] = n
        print(f"{n:>10}", end="")
    print()
    rows.append(row)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\nWrote {OUT}")
print(f"Margins E: {E}")
print("If n <= 20 -> 20 passes was enough. If n > 20 -> need more (or sensor-limited).")
