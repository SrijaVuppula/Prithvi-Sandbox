import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    from datasets import load_dataset
    repo = "ibm-nasa-geospatial/multi-temporal-crop-classification"
    print(f"[inspect] Loading first 3 samples from {repo} (streaming)...")
    dataset = load_dataset(repo, split="train", streaming=True)
    for i, sample in enumerate(dataset):
        print(f"\n--- Sample {i} ---")
        print(f"  Keys: {list(sample.keys())}")
        for k, v in sample.items():
            if hasattr(v, "shape"):
                arr = np.array(v)
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.2f}, max={arr.max():.2f}")
            elif isinstance(v, (list, tuple)):
                print(f"  {k}: list, len={len(v)}")
            else:
                print(f"  {k}: {v}")
        if i >= 2:
            break
    print("\n[inspect] Expected: shape like (H, W, 18) or (3, 6, 224, 224), values 0-10000.")

if __name__ == "__main__":
    main()
