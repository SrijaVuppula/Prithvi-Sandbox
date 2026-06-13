"""
Combined image-complexity score per chip (within the 500-chip study set).
Three components, each measures 'amount of stuff' differently:
  edge_density  - fraction of sharp edges (Canny)
  spatial_std   - how much pixel values vary across the image
  entropy       - how many distinct intensity levels (Shannon)
Each is normalised 0-1 across the 500 chips, then averaged into 'complexity'.
High = busy scene (hard reconstruction). Low = plain scene (easy).
"""
import os, glob, csv
import numpy as np
import rasterio
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy

CHIP_DIR = "multi_tile_generalization/study_chips_500"
OUT_CSV  = "multi_tile_generalization/block_masking_study/outputs/chip_complexity_v2.csv"
SUMMER_OFFSET = 6

def load_summer_rgb(path):
    with rasterio.open(path) as src:
        arr = src.read()
    s = arr[SUMMER_OFFSET:SUMMER_OFFSET+6].astype(np.float32)
    rgb = np.stack([s[2], s[1], s[0]], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

def raw_metrics(rgb):
    gray = rgb2gray(rgb)
    edge = float(canny(gray, sigma=1.0).mean())   # sharp-edge fraction
    std  = float(gray.std())                      # spatial variation
    ent  = float(shannon_entropy(gray))           # intensity diversity
    return edge, std, ent

def normalise(vals):
    v = np.array(vals, dtype=np.float64)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-12)

def main():
    chips = sorted(glob.glob(os.path.join(CHIP_DIR, "*_merged.tif")))
    chips = [c for c in chips if not os.path.basename(c).startswith("._")]
    print(f"Found {len(chips)} chips")

    names, edges, stds, ents = [], [], [], []
    for i, path in enumerate(chips):
        try:
            e, s, en = raw_metrics(load_summer_rgb(path))
            names.append(os.path.basename(path)); edges.append(e); stds.append(s); ents.append(en)
        except Exception as ex:
            print(f"SKIP {os.path.basename(path)}: {ex}")
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(chips)} done")

    ne, ns, nen = normalise(edges), normalise(stds), normalise(ents)
    complexity = (ne + ns + nen) / 3.0

    rows = sorted(zip(names, complexity, edges, stds, ents, ne, ns, nen),
                  key=lambda r: r[1], reverse=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chip","complexity","rank","edge_density","spatial_std","entropy",
                    "edge_norm","std_norm","entropy_norm"])
        for rank,(n,c,e,s,en,nne,nns,nnen) in enumerate(rows,1):
            w.writerow([n,f"{c:.4f}",rank,f"{e:.4f}",f"{s:.4f}",f"{en:.4f}",
                        f"{nne:.3f}",f"{nns:.3f}",f"{nnen:.3f}"])

    print(f"\nWrote {OUT_CSV}")
    print("\nTop 5 most complex (busy):")
    for n,c,*_ in rows[:5]:  print(f"  {c:.3f}  {n}")
    print("Bottom 8 (plainest):")
    for n,c,*_ in rows[-8:]: print(f"  {c:.3f}  {n}")
    for rank,(n,c,*_) in enumerate(rows,1):
        if n=="chip_091_324_merged.tif":
            print(f"\nchip_091_324 (our heterogeneous pick): complexity={c:.3f}, rank {rank}/{len(rows)}")

if __name__ == "__main__":
    main()
