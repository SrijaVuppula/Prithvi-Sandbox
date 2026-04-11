import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

CSV = Path("/home/myid/syv35378/Prithvi-Sandbox/outputs/results.csv")
OUT = Path("/home/myid/syv35378/Prithvi-Sandbox/outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# drop duplicate gap_type rows (regular==irregular with current data)
df = df[df["gap_type"] == "regular"].copy()

BACKBONE_ORDER = ["tiny", "100M", "300M", "600M"]
df["backbone"] = pd.Categorical(df["backbone"], categories=BACKBONE_ORDER, ordered=True)
df = df.sort_values("backbone")

METRICS = [
    ("mae",  "MAE (lower is better)",  False),
    ("psnr", "PSNR dB (higher is better)", True),
    ("ssim", "SSIM (higher is better)", True),
]

# ── Figure 1: backbone scaling by mask position ───────────────────────────────
for metric, ylabel, higher_better in METRICS:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, pos in zip(axes, ["middle", "endpoint"]):
        for t, ls in [(3, "--"), (4, "-")]:
            sub = df[(df["mask_position"] == pos) & (df["n_frames"] == t)]
            ax.plot(sub["backbone"], sub[metric], marker="o",
                    linestyle=ls, label=f"T={t}")
        ax.set_title(f"{pos} frame masked")
        ax.set_xlabel("Backbone")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Backbone scaling — {metric.upper()}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / f"backbone_scaling_{metric}.png", dpi=150)
    plt.close()
    print(f"Saved backbone_scaling_{metric}.png")

# ── Figure 2: middle vs endpoint by backbone (T=4) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
t4 = df[df["n_frames"] == 4]
for ax, (metric, ylabel, _) in zip(axes, METRICS):
    mid = t4[t4["mask_position"] == "middle"].set_index("backbone")[metric]
    end = t4[t4["mask_position"] == "endpoint"].set_index("backbone")[metric]
    x = range(len(BACKBONE_ORDER))
    w = 0.35
    ax.bar([i - w/2 for i in x], [mid[b] for b in BACKBONE_ORDER],
           width=w, label="middle")
    ax.bar([i + w/2 for i in x], [end[b] for b in BACKBONE_ORDER],
           width=w, label="endpoint")
    ax.set_xticks(list(x))
    ax.set_xticklabels(BACKBONE_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(metric.upper())
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Middle vs endpoint masking — T=4", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "position_comparison_T4.png", dpi=150)
plt.close()
print("Saved position_comparison_T4.png")

# ── Figure 3: T=3 vs T=4 context effect ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
mid = df[df["mask_position"] == "middle"]
for ax, (metric, ylabel, _) in zip(axes, METRICS):
    t3 = mid[mid["n_frames"] == 3].set_index("backbone")[metric]
    t4 = mid[mid["n_frames"] == 4].set_index("backbone")[metric]
    x = range(len(BACKBONE_ORDER))
    w = 0.35
    ax.bar([i - w/2 for i in x], [t3[b] for b in BACKBONE_ORDER],
           width=w, label="T=3")
    ax.bar([i + w/2 for i in x], [t4[b] for b in BACKBONE_ORDER],
           width=w, label="T=4")
    ax.set_xticks(list(x))
    ax.set_xticklabels(BACKBONE_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(metric.upper())
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Sequence length effect — middle masking", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "sequence_length_effect.png", dpi=150)
plt.close()
print("Saved sequence_length_effect.png")

# ── Table: best conditions per backbone ──────────────────────────────────────
best = df.loc[df.groupby("backbone")["psnr"].idxmax(),
              ["backbone","mask_position","n_frames","mae","psnr","ssim"]]
print("\nBest condition per backbone (by PSNR):")
print(best.to_string(index=False))