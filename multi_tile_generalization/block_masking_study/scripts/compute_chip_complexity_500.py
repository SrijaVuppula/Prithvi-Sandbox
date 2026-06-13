"""
Edge-density complexity score per chip.
High = more edges (buildings, roads, texture) = harder reconstruction.
Low  = homogeneous (farmland, water) = easier reconstruction.
"""
import os, glob, csv
import numpy as np
import rasterio
from skimage.feature import canny
from skimage.color import rgb2gray

CHIP_DIR = "multi_tile_generalization/study_chips_500"
OUT_CSV  = "multi_tile_generalization/block_masking_study/outputs/chip_complexity_500.csv"
SUMMER_OFFSET = 6

def load_summer_rgb(path):
    with rasterio.open(path) as src:
        arr = src.read()
    summer = arr[SUMMER_OFFSET:SUMMER_OFFSET+6].astype(np.float32)
    rgb = np.stack([summer[2], summer[1], summer[0]], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

def complexity_score(rgb):
    edges = canny(rgb2gray(rgb), sigma=1.0)
    return float(edges.mean())

def main():
    chips = sorted(glob.glob(os.path.join(CHIP_DIR, "*_merged.tif")))
    chips = [c for c in chips if not os.path.basename(c).startswith("._")]
    print(f"Found {len(chips)} chips")
    rows = []
    for i, path in enumerate(chips):
        try:
            rows.append((os.path.basename(path), complexity_score(load_summer_rgb(path))))
        except Exception as e:
            print(f"SKIP {os.path.basename(path)}: {e}")
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(chips)} done")
    rows.sort(key=lambda r: r[1], reverse=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip", "edge_density", "rank"])
        for rank, (name, score) in enumerate(rows, 1):
            w.writerow([name, f"{score:.6f}", rank])
    print(f"\nWrote {OUT_CSV}")
    print("\nTop 5 most complex:")
    for name, s in rows[:5]:  print(f"  {s:.4f}  {name}")
    print("Bottom 5 (most homogeneous):")
    for name, s in rows[-5:]: print(f"  {s:.4f}  {name}")
    target = "chip_003_062_merged.tif"
    for rank, (name, s) in enumerate(rows, 1):
        if name == target:
            pct = 100 * (1 - rank/len(rows))
            print(f"\n{target}: score={s:.4f}, rank {rank}/{len(rows)} (more complex than {pct:.0f}% of chips)")
            break

if __name__ == "__main__":
    main()
