"""Numbers for the paper, from paper_stats.csv (main run) or seed_stats.csv (10 seeds).
Usage: python scripts/paper_numbers.py            # self-check on the main run, then seed numbers
       python scripts/paper_numbers.py main|seeds # one mode only"""
import glob
from decimal import Decimal, ROUND_HALF_UP
import sys
import numpy as np
import pandas as pd
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
ORDER = ["tiny", "100M", "300M", "600M"]
LARGE = ORDER[1:]
RATIOS = [0.2, 0.4, 0.6, 0.8]

def q2(x):
    return Decimal(repr(round(x, 6))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def f2(x):
    return str(q2(x))

def cell(x, dag=False):
    q = q2(x)
    body = ("-" if q < 0 else "") + str(abs(q))
    if dag:
        return "$" + body + r"^{\dagger}$"
    return f"${body}$" if q < 0 else body

def inz(lo, hi):
    return lo <= 0 <= hi

def load_stats(mode):
    name = "paper_stats.csv" if mode == "main" else "seed_stats.csv"
    d = pd.read_csv(STUDY / "outputs_finetuned" / name)
    d["ratio"] = d.ratio.astype(float).round(2)
    return d.set_index(["backbone", "ratio"])

def train_cost(mode):
    out = {}
    for bb in ORDER:
        logs = STUDY / "outputs" / "finetune_logs"
        if mode == "main":
            fs = [str(logs / f"{bb}_finetune_cost.csv")]
        else:
            fs = sorted(glob.glob(str(logs / "seeds" / "seed*" / f"{bb}_finetune_cost.csv")))
        d = pd.concat([pd.read_csv(f) for f in fs])
        out[bb] = dict(h=d.elapsed_s.mean() / 3600, mj=d.energy_kj.mean() / 1000, w=d.avg_power_w.mean(), n=len(fs))
    return out

def energies():
    def tab(p):
        d = pd.read_csv(p)
        d = d[d.mask_type == "block"].copy()
        d["r"] = d.mask_ratio.astype(float).round(2)
        return {(b, r): float(v) / 1000.0 for b, r, v in zip(d.backbone, d.r, d.energy_mean)}  # file is in mJ
    return (tab(STUDY / "outputs" / "inference_energy.csv"),
            tab(STUDY / "outputs_finetuned" / "inference_energy_finetuned.csv"))

def breakeven(S, tc, Ez):
    rows, nos, undef = [], [], []
    for bb in ORDER:
        for r in RATIOS:
            ft = float(S.loc[(bb, r), "ft_contig_med"])
            alt = next((b for b in ORDER if float(S.loc[(b, r), "zs_contig_med"]) >= ft), None)
            if alt is None:
                undef.append((bb, int(r * 100)))
                continue
            if ORDER.index(alt) <= ORDER.index(bb):
                nos.append((bb, int(r * 100), round(float(S.loc[(bb, r), "zs_contig_med"]) - ft, 2), float(S.loc[(bb, r), "G_contiguous_med"])))
                continue
            saved = Ez[(alt, r)] - Ez[(bb, r)]
            if saved <= 0.05 * Ez[(alt, r)]:
                continue
            N = tc[bb]["mj"] * 1e6 / saved
            rows.append((bb, int(r * 100), alt, round(float(S.loc[(alt, r), "zs_contig_med"]) - ft, 2), saved / Ez[(alt, r)] * 100, N / 1e3))
    return rows, nos, undef

EXPECT = [("tiny", 20, "100M", 0.91, 188), ("tiny", 40, "100M", 0.79, 226), ("tiny", 60, "100M", 0.72, 230),
          ("tiny", 80, "100M", 0.60, 242), ("100M", 80, "300M", 0.28, 182), ("300M", 20, "600M", 0.27, 134),
          ("300M", 40, "600M", 0.55, 132), ("300M", 80, "600M", 0.03, 176)]

def run(mode):
    S, tc = load_stats(mode), train_cost(mode)
    Ez, Ef = energies()
    g = lambda col, b, r: float(S.loc[(b, r), col])
    rows, nos, undef = breakeven(S, tc, Ez)
    print(f"\n################ MODE: {mode} (n seeds per backbone = {[tc[b]['n'] for b in ORDER]}) ################")
    if mode == "main":
        got = [(b, r, a, e, round(N)) for b, r, a, e, s, N in rows]
        ok = len(got) == len(EXPECT) and all(x[:3] == y[:3] and abs(x[3] - y[3]) < 0.006 and abs(x[4] - y[4]) <= 1 for x, y in zip(got, EXPECT))
        print("SELF-CHECK vs current Table 4:", "MATCH" if ok else "MISMATCH", got if not ok else "")
        e20 = [round(Ez[(b, 0.2)], 2) for b in ORDER]; e80 = [round(Ez[(b, 0.8)], 2) for b in ORDER]
        print("Table 3 J/pass r=20:", e20, "(paper 1.77 3.54 7.54 19.76)  r=80:", e80, "(paper 1.69 3.06 6.86 16.15)")
        for b in ORDER:
            pc = [(Ef[(b, r)] - Ez[(b, r)]) / Ez[(b, r)] * 100 for r in RATIOS]
            print("  FT vs ZS %", b, round(min(pc), 2), "to", round(max(pc), 2))
    print("\n--- Table 1 rows ---")
    for b in ORDER:
        print(" & ".join([b] + [f2(g("zs_contig_med", b, r)) for r in RATIOS] + [f2(g("gap_zeroshot_med", b, r)) for r in RATIOS]
                         + [f2(g("gap_finetuned_med", b, r)) for r in RATIOS]
                         + [cell(g("dgap_med", b, r), inz(g("dgap_lo", b, r), g("dgap_hi", b, r))) for r in RATIOS]) + r" \\")
    print("\n--- Table 2 rows ---")
    for b in ORDER:
        print(" & ".join([b] + [cell(g("G_contiguous_med", b, r), inz(g("G_contiguous_lo", b, r), g("G_contiguous_hi", b, r))) for r in RATIOS]
                         + [cell(g("G_scattered_med", b, r), inz(g("G_scattered_lo", b, r), g("G_scattered_hi", b, r))) for r in RATIOS]) + r" \\")
    print("\n--- Table 4 rows ---")
    for b, r, a, e, s, N in rows:
        print(f"{b} & {r} & {a} & {e:.2f} & {s:.1f} & {N:.0f} " + r"\\")
    print("\n--- Table 3 training columns (time h, energy MJ) and power ---")
    for b in ORDER:
        print(b, f"{tc[b]['h']:.2f}", f"{tc[b]['mj']:.2f}", f"| power {tc[b]['w']:.1f} W")
    print("\n--- FACTS ---")
    tg = [g("G_contiguous_med", "tiny", r) for r in RATIOS]
    ts = [g("G_scattered_med", "tiny", r) for r in RATIOS]
    print("tiny G contig:", [f2(x) for x in tg], "| min % chips improving:", round(min(g("G_contiguous_pos", "tiny", r) for r in RATIOS) * 100, 1))
    print("tiny G scatt:", [f2(x) for x in ts], "| tiny Gc-Gs:", [f2(a - b) for a, b in zip(tg, ts)])
    cells = [(b, r) for b in LARGE for r in RATIOS]
    gc = [g("G_contiguous_med", b, r) for b, r in cells]
    print("large G contig min/max:", f2(min(gc)), f2(max(gc)))
    print("large G contig CI includes 0:", len([c for c in cells if inz(g("G_contiguous_lo", *c), g("G_contiguous_hi", *c))]), "of 12:",
          [(b, int(r * 100)) for b, r in cells if inz(g("G_contiguous_lo", b, r), g("G_contiguous_hi", b, r))])
    print("r=80 large: G", [f2(g("G_contiguous_med", b, 0.8)) for b in LARGE], "% improving", [round(g("G_contiguous_pos", b, 0.8) * 100) for b in LARGE],
          "CI excludes 0:", [not inz(g("G_contiguous_lo", b, 0.8), g("G_contiguous_hi", b, 0.8)) for b in LARGE])
    print("300M r=20 G contig:", f2(g("G_contiguous_med", "300M", 0.2)))
    lose = [(b, r) for b, r in cells if g("G_scattered_hi", b, r) < 0]
    near = [(b, int(r * 100), round(g("G_scattered_med", b, r), 3)) for b, r in cells if inz(g("G_scattered_lo", b, r), g("G_scattered_hi", b, r))]
    print("large G scatt: cells with CI below 0:", len(lose), "of 12 | losses range", f2(min(-g("G_scattered_med", *c) for c in lose)), "to", f2(max(-g("G_scattered_med", *c) for c in lose)), "| CI includes 0:", near)
    print("large cells with fall in scattered > contiguous gain:", sum(-g("G_scattered_med", b, r) > g("G_contiguous_med", b, r) for b, r in cells), "of 12")
    gz = [g("gap_zeroshot_med", b, r) for b in ORDER for r in RATIOS]
    gf = [g("gap_finetuned_med", b, r) for b in ORDER for r in RATIOS]
    print("ZS gap range:", f2(min(gz)), f2(max(gz)), "| min % chips positive:", round(min(g("gap_zeroshot_pos", b, r) for b in ORDER for r in RATIOS) * 100, 1))
    print("FT gap range:", f2(min(gf)), f2(max(gf)), "| min % chips positive:", round(min(g("gap_finetuned_pos", b, r) for b in ORDER for r in RATIOS) * 100, 1))
    print("ZS gap peak ratio:", {b: int(max(RATIOS, key=lambda r: g("gap_zeroshot_med", b, r)) * 100) for b in ORDER})
    print("FT gap peak ratio:", {b: int(max(RATIOS, key=lambda r: g("gap_finetuned_med", b, r)) * 100) for b in ORDER})
    dg = [(b, r, g("dgap_med", b, r), inz(g("dgap_lo", b, r), g("dgap_hi", b, r))) for b in ORDER for r in RATIOS]
    ex = [x for x in dg if not x[3]]
    print("change in gap: CI excludes 0 in", len(ex), "of 16 | reductions", f2(min(-x[2] for x in ex)), "to", f2(max(-x[2] for x in ex)),
          "| exceptions:", [(b, int(r * 100), round(v, 3)) for b, r, v, i in dg if i],
          "| tiny reductions:", [f2(-g("dgap_med", "tiny", r)) for r in RATIOS],
          "| max CI width:", round(max(g("dgap_hi", b, r) - g("dgap_lo", b, r) for b in ORDER for r in RATIOS), 3))
    cl = [(g("ft_contig_med", "tiny", r) - g("zs_contig_med", "tiny", r)) / (g("zs_contig_med", "100M", r) - g("zs_contig_med", "tiny", r)) * 100 for r in RATIOS]
    sv = [(1 - Ez[("tiny", r)] / Ez[("100M", r)]) * 100 for r in RATIOS]
    print("tiny closes % of gap to ZS 100M:", [round(x, 1) for x in cl], "| energy saving vs 100M %:", [round(x, 1) for x in sv])
    print("Fig 1 (r=40): closes", round(cl[1], 1), "% ; less energy", round(sv[1], 1), "% ; FT PSNR", [f2(g("ft_contig_med", b, 0.4)) for b in ORDER], "ZS PSNR", [f2(g("zs_contig_med", b, 0.4)) for b in ORDER])
    t4 = [x for x in rows if x[0] == "tiny"]
    o4 = [x for x in rows if x[0] != "tiny"]
    print("tiny: N* (10^3)", f"{min(x[5] for x in t4):.0f}-{max(x[5] for x in t4):.0f}", "| excess dB", f2(min(x[3] for x in t4)), f2(max(x[3] for x in t4)), "| saving %", f"{min(x[4] for x in t4):.1f}-{max(x[4] for x in t4):.1f}")
    print("larger: break-even cells", len(o4), "| N* (10^3)", f"{min(x[5] for x in o4):.0f}-{max(x[5] for x in o4):.0f}" if o4 else "-",
          "| energy ratio", f"{min(1 / (1 - x[4] / 100) for x in o4):.1f}-{max(1 / (1 - x[4] / 100) for x in o4):.1f}" if o4 else "-", "| excess", [x[3] for x in o4])
    print("no-saving cells:", len(nos), "| same-size ZS exceeds FT by", f"{min(x[2] for x in nos):.2f}-{max(x[2] for x in nos):.2f}" if nos else "-",
          "| of which median G contig > 0:", sum(x[3] > 0 for x in nos), "|", [(x[0], x[1], x[2], round(x[3], 3)) for x in nos], "| undefined:", undef)
    print("training: time h", [f"{tc[b]['h']:.2f}" for b in ORDER], "energy MJ", [f"{tc[b]['mj']:.2f}" for b in ORDER], "power W", [f"{tc[b]['w']:.0f}" for b in ORDER])
    fe = [tc[ORDER[i + 1]]["mj"] / tc[ORDER[i]]["mj"] for i in range(3)]
    ft_ = [tc[ORDER[i + 1]]["h"] / tc[ORDER[i]]["h"] for i in range(3)]
    print("per-step factors: energy", [round(float(x), 2) for x in fe], "time", [round(float(x), 2) for x in ft_])
    eq = [tc[b]["mj"] * 1e6 / Ez[(b, 0.2)] for b in ORDER]
    print("training energy = forward passes at r=20%:", [round(x / 1e3) for x in eq], "thousand")
    if "G_contiguous_seedsd" in S.columns:
        sd_c = [g("G_contiguous_seedsd", b, r) for b in ORDER for r in RATIOS]
        sd_s = [g("G_scattered_seedsd", b, r) for b in ORDER for r in RATIOS]
        sd_d = [g("dgap_seedsd", b, r) for b in ORDER for r in RATIOS]
        print("run-to-run SD of median G: contiguous max", f2(max(sd_c)), "scattered max", f2(max(sd_s)), "| change in gap max", f2(max(sd_d)))

if __name__ == "__main__":
    modes = [sys.argv[1]] if len(sys.argv) > 1 else ["main", "seeds"]
    for m in modes:
        run(m)
