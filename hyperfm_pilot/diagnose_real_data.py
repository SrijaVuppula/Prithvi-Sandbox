"""
Two things to confirm with real data/files before building the real-data
loader, rather than guessing:
  1. pace_band_wavelengths.py's exact arrays -- spectral_band_adapter.py
     currently has a PLACEHOLDER for the 9 discrete SWIR centers; this file
     (from the Aug 24 wavelength-validation session) has the real ones.
  2. One real .npy tile's actual shape/dtype/value range -- confirms the
     96x96x291 float32 assumption and the 96 vs Prithvi's 224 patch-size
     mismatch before deciding how to handle it.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "hyperfm_pilot"))

print("=" * 60)
print("pace_band_wavelengths.py -- real arrays:")
try:
    import pace_band_wavelengths as pw
    for name in ["blue_wavelength", "red_wavelength", "SWIR_wavelength"]:
        arr = getattr(pw, name, None)
        if arr is None:
            print(f"  {name}: NOT FOUND as a module-level attribute")
            continue
        arr = np.asarray(arr)
        print(f"  {name}: len={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}")
        print(f"    values: {list(np.round(arr, 2))}")
except Exception as e:
    print(f"import FAILED: {type(e).__name__}: {e}")
    print("paste back `cat hyperfm_pilot/pace_band_wavelengths.py` instead")

print("\n" + "=" * 60)
print("Real tile check:")
data_dir = Path.home() / "Prithvi" / "hyperfm_pilot" / "cvpr_dataset" / "hsi"
npy_files = list(data_dir.glob("*.npy"))
print(f"found {len(npy_files)} .npy files in {data_dir}")
if npy_files:
    tile = np.load(npy_files[0])
    print(f"sample file: {npy_files[0].name}")
    print(f"shape: {tile.shape}, dtype: {tile.dtype}")
    print(f"min: {tile.min():.4f}, max: {tile.max():.4f}, mean: {tile.mean():.4f}")
    print(f"any NaN: {np.isnan(tile).any()}, any Inf: {np.isinf(tile).any()}")
