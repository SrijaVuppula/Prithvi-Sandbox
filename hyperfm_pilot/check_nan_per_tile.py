import numpy as np
import glob
import os

files = sorted(glob.glob('cvpr_dataset/**/*.npy', recursive=True))
print(f"{len(files)} tiles\n")

clean, dirty = [], []
for f in files:
    arr = np.load(f)
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    pct_nan = 100 * n_nan / arr.size
    tag = os.path.basename(f)
    if n_nan == 0 and n_inf == 0:
        clean.append(tag)
    else:
        dirty.append((tag, n_nan, n_inf, pct_nan))
        print(f"DIRTY  {tag}: nan={n_nan} ({pct_nan:.1f}%)  inf={n_inf}")

print(f"\nClean tiles: {len(clean)}/{len(files)}")
print(f"Dirty tiles: {len(dirty)}/{len(files)}")
