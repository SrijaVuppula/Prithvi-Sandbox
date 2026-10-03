"""Cost tables for the paper: inference energy zs vs ft (contiguous=block, forward-order runs),
one-time training cost, and break-even N* per the protocol definition."""
import pandas as pd
from pathlib import Path

S = Path(__file__).resolve().parent.parent
BB = ["tiny", "100M", "300M", "600M"]
RATIOS = [0.2, 0.4, 0.6, 0.8]
pd.set_option("display.width", 250, "display.max_columns", 50)

zs = pd.read_csv(S / "outputs" / "inference_energy.csv")
ft = pd.read_csv(S / "outputs_finetuned" / "inference_energy_finetuned.csv")
def blk(d):
    return d[d.mask_type == "block"].set_index(["backbone", "mask_ratio"])["energy_mean"]
Ez, Ef = blk(zs), blk(ft)  # mJ per forward pass

print("\n== Inference energy per pass, contiguous (mJ)")
rows = []
for bb in BB:
    for r in RATIOS:
        rows.append({"backbone": bb, "ratio": r, "zs_mJ": round(Ez[(bb, r)], 1),
                     "ft_mJ": round(Ef[(bb, r)], 1),
                     "ft_vs_zs_pct": round(100 * (Ef[(bb, r)] / Ez[(bb, r)] - 1), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== Zero-shot energy ratio between consecutive sizes")
for r in RATIOS:
    print(r, [round(Ez[(BB[i + 1], r)] / Ez[(BB[i], r)], 2) for i in range(3)])

print("\n== One-time training cost (raw columns from *_finetune_cost.csv)")
cost = {bb: pd.read_csv(S / "outputs" / "finetune_logs" / f"{bb}_finetune_cost.csv").iloc[0] for bb in BB}
print(pd.DataFrame(cost).T.to_string())

st = pd.read_csv(S / "outputs_finetuned" / "paper_stats.csv").set_index(["backbone", "ratio"])
print("\n== Break-even N* (protocol definition; energy_saved must exceed 5% of E_zs)")
out = []
for b in BB:
    e_train_mJ = float(cost[b]["energy_kj"]) * 1e6
    for r in RATIOS:
        target = st.loc[(b, r), "ft_contig_med"]
        row = {"ft_backbone": b, "ratio": r, "ft_psnr": target,
               "G_contig": st.loc[(b, r), "G_contiguous_med"]}
        z = next((x for x in BB if st.loc[(x, r), "zs_contig_med"] >= target), None)
        if z is None:
            row["note"] = "no zs qualifies" + ("; N* undefined (largest)" if b == "600M" else "; use 600M (upper bound)")
            z = None if b == "600M" else "600M"
        if z is not None:
            row["zs_alt"] = z
            row["overshoot_dB"] = round(st.loc[(z, r), "zs_contig_med"] - target, 2)
            if BB.index(z) <= BB.index(b):
                row.setdefault("note", "no saving (own/smaller zs qualifies)")
            else:
                saved = Ez[(z, r)] - Ez[(b, r)]
                row["saved_pct_of_Ezs"] = round(100 * saved / Ez[(z, r)], 1)
                if saved > 0.05 * Ez[(z, r)]:
                    row["N_star"] = f"{e_train_mJ / saved:,.0f}"
                else:
                    row.setdefault("note", "saving below 5% floor")
        out.append(row)
print(pd.DataFrame(out).fillna("").to_string(index=False))
