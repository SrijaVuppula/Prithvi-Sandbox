"""
_style.py — shared plotting conventions for the corrected block masking figures.

Uses the external ~/Prithvi/plot_style.py when present (Okabe-Ito, apply_style,
BACKBONE_COLORS). Falls back to an inlined Okabe-Ito palette if absent.
"""
import os, sys

sys.path.insert(0, os.path.expanduser("~/Prithvi"))

BACKBONES = ["tiny", "100M", "300M", "600M"]
RATIOS    = [0.20, 0.40, 0.60, 0.80]
MARKERS   = {"tiny": "o", "100M": "s", "300M": "^", "600M": "D"}

_OKABE = {"tiny": "#0072B2", "100M": "#E69F00", "300M": "#009E73", "600M": "#CC79A7"}


def get_style():
    """Return (apply_style_fn, backbone_colors). Prefers the shared file."""
    try:
        from plot_style import apply_style, BACKBONE_COLORS  # type: ignore
        colors = {b: BACKBONE_COLORS.get(b, _OKABE[b]) for b in BACKBONES}
        return apply_style, colors
    except Exception:
        import matplotlib.pyplot as plt
        def _fallback():
            plt.rcParams.update({
                "figure.dpi": 120, "savefig.dpi": 200,
                "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
                "legend.frameon": False,
            })
        return _fallback, dict(_OKABE)
