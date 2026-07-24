"""
Ranks all study chips by spring->summer->fall pixel variation to find
candidates for Task 2 (high seasonal-change chips).
"""
import os
import glob
import numpy as np
import rasterio

CHIP_DIR = "multi_tile_generalization/study_chips_500"
PATTERN = os.path.join(CHIP_DIR, "*merged.tif")

def load_chip(path):
    with rasterio.open(path) as src:
        arr = src.read()  # (18, 224, 224) raw HLS scale
    arr = arr.reshape(3, 6, 224, 224).astype(np.float32) / 10000.0  # (T, C, H, W)
    return arr

def seasonal_variation(chip):
    spring, summer, fall = chip[0], chip[1], chip[2]
    d_ss = np.abs(summer - spring).mean()
    d_sf = np.abs(fall - summer).mean()
    d_spf = np.abs(fall - spring).mean()
    return d_ss, d_sf, d_spf

paths = sorted(glob.glob(PATTERN))
print(f"Found {len(paths)} chips matching {PATTERN}")

if not paths:
    print("No chips found — check the path/pattern above against the ls/find output.")
else:
    results = []
    for p in paths:
        try:
            chip = load_chip(p)
        except Exception as e:
            print(f"SKIP {os.path.basename(p)}: {e}")
            continue
        d_ss, d_sf, d_spf = seasonal_variation(chip)
        results.append((os.path.basename(p), d_ss, d_sf, d_spf, d_ss + d_sf))

    results.sort(key=lambda r: r[-1], reverse=True)

    print(f"\nTop 15 by (spring->summer + summer->fall) pixel variation:")
    print(f"{'chip':>28} | {'spr->sum':>9} {'sum->fall':>10} {'spr->fall':>10} | {'total':>8}")
    for r in results[:15]:
        print(f"{r[0]:>28} | {r[1]:9.4f} {r[2]:10.4f} {r[3]:10.4f} | {r[4]:8.4f}")
