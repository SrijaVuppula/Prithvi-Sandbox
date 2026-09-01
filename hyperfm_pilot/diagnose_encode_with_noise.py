"""
Diagnostic: check _encode_with_noise's source for its own no_grad/inference
guards before trusting it as a gradient-safe building block. run_masked_forward
itself is @torch.no_grad()-decorated (confirmed), but it calls
_encode_with_noise(...) then model.decoder(...) as two separate steps --
if _encode_with_noise doesn't have its own independent no_grad guard, those
same two calls, made directly (bypassing the decorated wrapper function),
should be gradient-safe.
"""
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path.home() / "Prithvi" / "Prithvi-Sandbox"
sys.path.insert(0, str(REPO_ROOT / "patch_masking_study"))

import terratorch_loader  # noqa: E402

print("=" * 60)
print("_encode_with_noise source:")
print(inspect.getsource(terratorch_loader._encode_with_noise))

print("\n" + "=" * 60)
print("Checking for no_grad/inference_mode decorators on both functions:")
for fn_name in ["run_masked_forward", "_encode_with_noise"]:
    fn = getattr(terratorch_loader, fn_name)
    print(f"{fn_name}: __wrapped__ present = {hasattr(fn, '__wrapped__')}")
