"""
Convert every .npy error map in a directory to a viewable .png, saved
alongside the source .npy (same directory, same basename). Also saves a
combined grid figure. Shared color scale (vmax = max across all files)
so panels are directly comparable.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                         help="directory containing .npy error maps")
    args = parser.parse_args()
    d = Path(args.dir)
    npy_files = sorted(d.glob("*.npy"))
    assert npy_files, f"no .npy files found in {d}"

    arrs = {f.stem: np.load(f) for f in npy_files}
    vmax = max(a.max() for a in arrs.values())

    for name, arr in arrs.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(arr, vmin=0, vmax=vmax, cmap="inferno")
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8, label="abs error")
        out_path = d / f"{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")

    n = len(arrs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for i, (name, arr) in enumerate(arrs.items()):
        ax = axes[i // ncols, i % ncols]
        im = ax.imshow(arr, vmin=0, vmax=vmax, cmap="inferno")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="abs error")
    grid_path = d / "errormap_grid.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {grid_path}")

if __name__ == "__main__":
    main()
