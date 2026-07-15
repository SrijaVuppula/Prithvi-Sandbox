import pandas as pd

IN_CSV   = "multi_tile_generalization/block_masking_study/outputs/energy_FIXED_warmup_order_10_20_40_60_80_allBB.csv"
OUT_RB   = "multi_tile_generalization/block_masking_study/outputs/energy_jumps_random_vs_block_FIXED.csv"
OUT_STEP = "multi_tile_generalization/block_masking_study/outputs/energy_jumps_ratio_steps_FIXED.csv"

df = pd.read_csv(IN_CSV)

def jump(a, b):
    lo, hi = sorted([a, b])
    return hi / lo if lo > 0 else float("nan")

# Jump 1: random vs block, same backbone+ratio
piv = df.pivot_table(index=["backbone", "mask_ratio"],
                      columns="mask_type",
                      values="energy_mean").reset_index()
piv["jump_random_vs_block"] = piv.apply(lambda r: jump(r["random"], r["block"]), axis=1)
piv = piv.sort_values(["backbone", "mask_ratio"])
piv.to_csv(OUT_RB, index=False)

# Jump 2: ratio-step jump within same backbone+mask_type (10->20->40->60->80)
d = df.sort_values(["backbone", "mask_type", "mask_ratio"]).copy()
d["energy_prev"] = d.groupby(["backbone", "mask_type"])["energy_mean"].shift(1)
d["ratio_prev"]  = d.groupby(["backbone", "mask_type"])["mask_ratio"].shift(1)
d["jump_step"] = d.apply(
    lambda r: jump(r["energy_mean"], r["energy_prev"]) if pd.notna(r["energy_prev"]) else None,
    axis=1
)
d[["backbone","mask_type","ratio_prev","mask_ratio","energy_prev","energy_mean","jump_step"]].to_csv(OUT_STEP, index=False)

print("=== Random vs Block jump per backbone x ratio (FIXED data) ===")
print(piv.to_string(index=False))
print("\n=== Ratio-step jump per backbone x mask_type (FIXED data) ===")
print(d[["backbone","mask_type","ratio_prev","mask_ratio","jump_step"]].dropna().to_string(index=False))
