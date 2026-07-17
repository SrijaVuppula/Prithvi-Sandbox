"""
Computes energy comparison tables from the doubly-corrected inference_energy.csv
(GPU-warmup fix + masking fix, both applied).
Replaces the July 12 version, which unknowingly read pre-masking-fix data.
Outputs:
  - energy_jumps_random_vs_block.csv : block vs random energy ratio per backbone/ratio
  - energy_jumps_ratio_steps.csv     : step-to-step energy ratio per backbone, with biggest jump flagged
"""
import pandas as pd
from pathlib import Path

IN_FILE = Path("outputs/inference_energy.csv")
OUT_DIR = Path("outputs")

df = pd.read_csv(IN_FILE)

# --- Block vs random ratio, per backbone/ratio ---
piv = df.pivot_table(index=["backbone", "mask_ratio"], columns="mask_type", values="energy_mean")
piv["ratio"] = piv[["block", "random"]].max(axis=1) / piv[["block", "random"]].min(axis=1)
piv = piv.round(4).reset_index()
piv.to_csv(OUT_DIR / "energy_jumps_random_vs_block.csv", index=False)
print("=== Block vs Random energy ratio ===")
print(piv.to_string(index=False))

# --- Per-backbone step-to-step ratio, biggest jump flagged ---
avg = df.groupby(["backbone", "mask_ratio"])["energy_mean"].mean().reset_index()
rows = []
for bb in sorted(avg["backbone"].unique()):
    sub = avg[avg["backbone"] == bb].sort_values("mask_ratio").reset_index(drop=True)
    sub["step_ratio"] = sub["energy_mean"].shift(1) / sub["energy_mean"]
    sub["backbone"] = bb
    rows.append(sub)
steps = pd.concat(rows, ignore_index=True)
steps = steps.round(4)
steps.to_csv(OUT_DIR / "energy_jumps_ratio_steps.csv", index=False)

print("\n=== Per-backbone step sizes (biggest jump flagged) ===")
for bb in sorted(steps["backbone"].unique()):
    sub = steps[steps["backbone"] == bb].dropna(subset=["step_ratio"])
    biggest = sub.loc[sub["step_ratio"].idxmax()]
    print(f"\n{bb}: biggest jump ends at ratio={biggest['mask_ratio']} (step_ratio={biggest['step_ratio']:.3f})")
    print(steps[steps["backbone"] == bb][["mask_ratio", "energy_mean", "step_ratio"]].to_string(index=False))

print(f"\nWrote: {OUT_DIR/'energy_jumps_random_vs_block.csv'}")
print(f"Wrote: {OUT_DIR/'energy_jumps_ratio_steps.csv'}")
