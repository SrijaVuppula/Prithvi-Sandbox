"""
Diagnostic: print run_masked_forward's actual source. The forward pass
through both adapters succeeds and produces correct shapes, but backward()
fails with "does not require grad and does not have a grad_fn" -- almost
certainly a torch.no_grad() or torch.inference_mode() context somewhere in
this function, since it's only ever been used for zero-shot evaluation
before now, never training.
"""
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))

import terratorch_loader  # noqa: E402

print("=" * 60)
print("run_masked_forward source:")
print(inspect.getsource(terratorch_loader.run_masked_forward))
