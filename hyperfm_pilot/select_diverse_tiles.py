"""
Samples a larger candidate pool from the full 4,250-tile manifest,
extracts them, scores each by overall tile variance (a simple proxy
for "how different is this scene" -- flat cloud tops score low,
mixed surface/cloud-edge scenes score high), then picks 20 tiles
spread evenly across that score range so the reconstruction test
covers genuinely different scene types, not just 20 similar ones.
"""
import subprocess, random
from pathlib import Path
import numpy as np

MANIFEST = Path("hsi_manifest.txt")
TAR_PATH = Path.home() / "Prithvi" / "hyperfm_pilot" / "PACE_CLD_CLDMASK.tar"
N_CANDIDATES = 500
N_SELECT = 100
SEED = 123

rng = random.Random(SEED)
all_tiles = [l.strip() for l in MANIFEST.read_text().splitlines() if l.strip()]
candidates = rng.sample(all_tiles, N_CANDIDATES)

candidate_list_file = Path("candidate_tiles.txt")
candidate_list_file.write_text("\n".join(candidates) + "\n")

print(f"Extracting {N_CANDIDATES} candidate tiles from tar (single pass through the archive)...")
subprocess.run(
    ["tar", "-xf", str(TAR_PATH), "-C", ".", "-T", str(candidate_list_file)],
    check=True,
)

print("Scoring tiles by overall variance (diversity proxy)...")
scores = []
for tf in candidates:
    p = Path(tf)
    if not p.exists():
        print(f"  MISSING after extraction: {tf}")
        continue
    tile = np.load(p).astype(np.float64)
    score = float(np.nanstd(tile))
    mean_val = float(np.nanmean(tile))
    scores.append((tf, score, mean_val))

scores.sort(key=lambda x: x[1])
n = len(scores)
print(f"Scored {n}/{N_CANDIDATES} tiles successfully.")

pick_idx = sorted(set(np.linspace(0, n - 1, N_SELECT).round().astype(int).tolist()))
while len(pick_idx) < N_SELECT:
    remaining = [i for i in range(n) if i not in pick_idx]
    pick_idx.append(remaining[len(pick_idx) % len(remaining)])
pick_idx = sorted(pick_idx[:N_SELECT])
selected = [scores[i] for i in pick_idx]

with open("hsi_diverse_100.txt", "w") as f:
    for tf, _, _ in selected:
        f.write(tf + "\n")

with open("hsi_diverse_100_scores.csv", "w") as f:
    f.write("tile,variance_score,mean_radiance\n")
    for tf, sc, mv in selected:
        f.write(f"{tf},{sc:.5f},{mv:.5f}\n")

print(f"\nSelected {len(selected)} tiles spanning variance range "
      f"{selected[0][1]:.4f} to {selected[-1][1]:.4f}")
print("Saved: hsi_diverse_100.txt, hsi_diverse_100_scores.csv")

selected_set = {tf for tf, _, _ in selected}
removed = 0
for tf in candidates:
    if tf not in selected_set:
        p = Path(tf)
        if p.exists():
            p.unlink()
            removed += 1
print(f"Cleaned up {removed} non-selected extracted tiles.")
