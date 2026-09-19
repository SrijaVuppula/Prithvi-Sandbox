"""
select_diverse_400.py

Extends hsi_diverse_100.txt to 400 tiles for the E=0.5dB power-analysis
rerun (n=(z*sigma/E)^2 with the ~1dB margin measured on n=100 -> ~400
needed for E=0.5dB). All 4,250 HyperFM250K tiles are already extracted
to cvpr_dataset/hsi/, so this is pure CPU scoring -- no tar extraction.

Scores every tile not already in hsi_diverse_100.txt by overall variance
(same diversity proxy as select_diverse_tiles.py), picks 300 new ones
spread evenly across that score range, and writes hsi_diverse_400.txt
as the existing 100 + the 300 new (so results_zeroshot/ratio_sweep_results_v2.csv
rows for the original 100 can be reused rather than rerun).
"""
from pathlib import Path
import numpy as np

MANIFEST = Path("hsi_manifest.txt")
EXISTING = Path("hsi_diverse_100.txt")
HSI_DIR = Path("cvpr_dataset/hsi")
N_NEW = 300
SEED = 123

existing = set(l.strip() for l in EXISTING.read_text().splitlines() if l.strip())
all_tiles = [l.strip() for l in MANIFEST.read_text().splitlines() if l.strip()]

# candidate pool: every manifest tile not already selected, that's actually
# present on disk (should be all 4,250, but check rather than assume)
candidates = []
missing = 0
for tf in all_tiles:
    if tf in existing:
        continue
    p = Path(tf)
    if not p.exists():
        missing += 1
        continue
    candidates.append(tf)

print(f"Manifest: {len(all_tiles)}  existing: {len(existing)}  "
      f"candidates on disk: {len(candidates)}  missing from disk: {missing}")

print(f"Scoring {len(candidates)} candidate tiles by variance...")
scores = []
for tf in candidates:
    tile = np.load(Path(tf)).astype(np.float64)
    scores.append((tf, float(np.nanstd(tile))))

scores.sort(key=lambda x: x[1])
n = len(scores)
if n < N_NEW:
    raise RuntimeError(f"Only {n} candidates available, need {N_NEW}")

pick_idx = sorted(set(np.linspace(0, n - 1, N_NEW).round().astype(int).tolist()))
while len(pick_idx) < N_NEW:
    remaining = [i for i in range(n) if i not in pick_idx]
    pick_idx.append(remaining[len(pick_idx) % len(remaining)])
pick_idx = sorted(pick_idx[:N_NEW])
new_selected = [scores[i][0] for i in pick_idx]

combined = sorted(existing) + new_selected
with open("hsi_diverse_400.txt", "w") as f:
    for tf in combined:
        f.write(tf + "\n")

print(f"\nSelected {len(new_selected)} new tiles (variance range "
      f"{scores[pick_idx[0]][1]:.4f} to {scores[pick_idx[-1]][1]:.4f})")
print(f"Total in hsi_diverse_400.txt: {len(combined)} "
      f"({len(existing)} existing + {len(new_selected)} new)")
