"""Print the DATA block of fig1_overview_v2.py from the 10-seed results.
Reads outputs_finetuned/seed_stats.csv and outputs/inference_energy.csv (energy_mean is in mJ)."""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
BB = ["tiny", "100M", "300M", "600M"]
RATIOS = [0.2, 0.4, 0.6, 0.8]

S = pd.read_csv(STUDY / "outputs_finetuned" / "seed_stats.csv")
S["ratio"] = S.ratio.astype(float).round(2)
S = S.set_index(["backbone", "ratio"])
E = pd.read_csv(STUDY / "outputs" / "inference_energy.csv")
E = E[E.mask_type == "block"].copy()
E["r"] = E.mask_ratio.astype(float).round(2)
E = E.set_index(["backbone", "r"])

def q2(x):
    return Decimal(repr(round(float(x), 6))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def tstr(x):
    q = q2(x)
    return ('M + "' + str(abs(q)) + '"') if q < 0 else '"' + str(q) + '"'

def inz(b, r, name):
    return int(S.loc[(b, r), name + "_lo"] <= 0 <= S.loc[(b, r), name + "_hi"])

print("# ------------------------------------------------------------------ DATA (10-seed means)")
print('BB = ["tiny", "100M", "300M", "600M"]')
print("ENERGY_J = {" + ", ".join(f'"{b}": {E.loc[(b, 0.4), "energy_mean"] / 1000:.4f}' for b in BB) + "}   # J / pass, r = 40%")
print("PSNR_ZS = {" + ", ".join(f'"{b}": {S.loc[(b, 0.4), "zs_contig_med"]:.2f}' for b in BB) + "}")
print("PSNR_FT = {" + ", ".join(f'"{b}": {S.loc[(b, 0.4), "ft_contig_med"]:.2f}' for b in BB) + "}")
for var, name in [("G_CONT", "G_contiguous"), ("G_SCAT", "G_scattered")]:
    rows = ["[" + ", ".join(f"{S.loc[(b, r), name + '_med']:.3f}".replace("0.", ".", 1) if abs(S.loc[(b, r), name + '_med']) < 1 else f"{S.loc[(b, r), name + '_med']:.3f}" for r in RATIOS) + "]" for b in BB]
    print(f"{var} = np.array([" + ", ".join(rows) + "])")
print('M = "\\u2212"')
for var, name in [("T_CONT", "G_contiguous"), ("T_SCAT", "G_scattered")]:
    print(f"{var} = [" + ", ".join("[" + ", ".join(tstr(S.loc[(b, r), name + '_med']) for r in RATIOS) + "]" for b in BB) + "]")
for var, name in [("D_CONT", "G_contiguous"), ("D_SCAT", "G_scattered")]:
    print(f"{var} = [" + ", ".join("[" + ", ".join(str(inz(b, r, name)) for r in RATIOS) + "]" for b in BB) + "]")
cl = (S.loc[("tiny", 0.4), "ft_contig_med"] - S.loc[("tiny", 0.4), "zs_contig_med"]) / (S.loc[("100M", 0.4), "zs_contig_med"] - S.loc[("tiny", 0.4), "zs_contig_med"]) * 100
sv = (1 - E.loc[("tiny", 0.4), "energy_mean"] / E.loc[("100M", 0.4), "energy_mean"]) * 100
mx = max(abs(S.loc[(b, 0.4), "ft_contig_med"] - S.loc[(b, 0.4), "zs_contig_med"]) for b in BB[1:])
print(f"\n# annotation values: tiny closes {cl:.1f}% of the gap to 100M at {sv:.1f}% less energy; larger backbones within +/-{mx:.2f} dB")
