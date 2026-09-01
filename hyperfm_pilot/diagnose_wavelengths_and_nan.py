"""
Follow-up diagnostic: my guessed attribute names for pace_band_wavelengths.py
were wrong (real names: BAND_ORIGIN, BAND_WAVELENGTH_NM, BLUE_WAVELENGTH_NM,
RED_WAVELENGTH_NM, SWIR_WAVELENGTH_NM). Also characterize the NaN pattern in
real tiles -- HyperFM250K is described as >60% cloud-covered, so NaNs are
likely cloud-masked pixels, not a data error, but need to know the actual
extent/pattern before deciding how to handle them in a loader.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

print("=" * 60)
print("pace_band_wavelengths.py -- actual module contents:")
import pace_band_wavelengths as pw  # noqa: E402
names = [n for n in dir(pw) if not n.startswith("_")]
print(names)
for n in names:
    val = getattr(pw, n)
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val)
        if np.issubdtype(arr.dtype, np.number):
            print(f"  {n}: len={len(arr)}, dtype={arr.dtype}, min={arr.min():.2f}, max={arr.max():.2f}")
            if len(arr) <= 20:
                print(f"    values: {list(np.round(arr, 3))}")
            else:
                print(f"    first 5: {list(np.round(arr[:5], 3))}  last 5: {list(np.round(arr[-5:], 3))}")
        else:
            print(f"  {n}: len={len(arr)}, dtype={arr.dtype}, sample={arr[:5]}")

print("\n" + "=" * 60)
print("NaN pattern across several real tiles:")
data_dir = Path.home() / "Prithvi" / "hyperfm_pilot" / "cvpr_dataset" / "hsi"
npy_files = sorted(data_dir.glob("*.npy"))[:5]
for f in npy_files:
    tile = np.load(f)  # (96, 96, 291)
    nan_mask = np.isnan(tile)
    pct_nan_overall = 100 * nan_mask.mean()
    pixel_all_nan = nan_mask.all(axis=-1)
    pixel_any_nan = nan_mask.any(axis=-1)
    pct_pixels_fully_nan = 100 * pixel_all_nan.mean()
    pct_pixels_any_nan = 100 * pixel_any_nan.mean()
    valid = tile[~nan_mask]
    print(f"{f.name}:")
    print(f"  overall NaN%: {pct_nan_overall:.1f} | pixels fully-NaN: {pct_pixels_fully_nan:.1f}% "
          f"| pixels with ANY NaN band: {pct_pixels_any_nan:.1f}%")
    if valid.size:
        print(f"  valid-value range: [{valid.min():.4f}, {valid.max():.4f}]")
