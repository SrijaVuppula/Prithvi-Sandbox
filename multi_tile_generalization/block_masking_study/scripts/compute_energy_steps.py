"""
compute_energy_steps.py
-----------------------
Energy/time change between consecutive mask ratios, per backbone.

Block and random encode identical token counts at a matched summer count, so
their energies are identical by construction — there is no geometry effect to
measure. This script therefore averages over mask_type and reports only the
ratio-to-ratio steps, which is where the real (sub-proportional, stepped)
behaviour lives.

Reads outputs/inference_energy.csv -> outputs/energy_ratio_steps.csv
"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
df = pd.read_csv(OUT_DIR / "inference_energy.csv")

g = (df.groupby(["backbone", "mask_ratio"])
       .agg(tokens_encoded=("tokens_encoded", "first"),
            time_ms=("time_mean", "mean"),
            energy_mj=("energy_mean", "mean"))
       .reset_index())

rows = []
for bb in ["tiny", "100M", "300M", "600M"]:
    sub = g[g["backbone"] == bb].sort_values("mask_ratio").reset_index(drop=True)
    for i in range(1, len(sub)):
        a, b = sub.loc[i-1], sub.loc[i]
        rows.append({
            "backbone": bb,
            "step": f"{int(a.mask_ratio*100)}->{int(b.mask_ratio*100)}%",
            "tokens_from": int(a.tokens_encoded), "tokens_to": int(b.tokens_encoded),
            "tokens_pct_drop": round((a.tokens_encoded - b.tokens_encoded) / a.tokens_encoded * 100, 2),
            "time_pct_drop":   round((a.time_ms - b.time_ms) / a.time_ms * 100, 2),
            "energy_pct_drop": round((a.energy_mj - b.energy_mj) / a.energy_mj * 100, 2),
        })

steps = pd.DataFrame(rows)
out = OUT_DIR / "energy_ratio_steps.csv"
steps.to_csv(out, index=False)
print(steps.to_string(index=False))
print(f"\nWrote {out}")

print("\nFull sweep (10% -> 80%):")
for bb in ["tiny", "100M", "300M", "600M"]:
    sub = g[g["backbone"] == bb].sort_values("mask_ratio")
    lo, hi = sub.iloc[0], sub.iloc[-1]
    print(f"  {bb:>5}: tokens -{(lo.tokens_encoded-hi.tokens_encoded)/lo.tokens_encoded*100:5.1f}%  "
          f"energy -{(lo.energy_mj-hi.energy_mj)/lo.energy_mj*100:5.1f}%")
