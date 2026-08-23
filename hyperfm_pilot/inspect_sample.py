import numpy as np
import glob

files = sorted(glob.glob('cvpr_dataset/**/*.npy', recursive=True))
print("Files found:", files)

for f in files[:1]:
    arr = np.load(f)
    print(f"\n{f}")
    print("shape:", arr.shape, "dtype:", arr.dtype)
    print("min/max:", arr.min(), arr.max())
    print("nan count:", np.isnan(arr).sum(), "/ total", arr.size)
    print("inf count:", np.isinf(arr).sum())

    n_bands = arr.shape[-1]
    corrs = []
    for i in range(0, n_bands - 1, 10):
        a = arr[:, :, i].flatten()
        b = arr[:, :, i + 1].flatten()
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        corrs.append(np.corrcoef(a, b)[0, 1])
    print(f"adjacent-band corr (sampled every 10): mean={np.mean(corrs):.4f}, min={np.min(corrs):.4f}, max={np.max(corrs):.4f}")

    a, b = arr[:, :, 0].flatten(), arr[:, :, -1].flatten()
    if not (np.isnan(a).any() or np.isnan(b).any()):
        print(f"far-band corr (band 0 vs {n_bands-1}): {np.corrcoef(a, b)[0, 1]:.4f}")
