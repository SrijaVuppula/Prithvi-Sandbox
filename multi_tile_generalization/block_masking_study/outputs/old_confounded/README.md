# Superseded: confounded block-vs-random artifacts

Everything here predates the masking fix (July 2026) and is kept as the record
of the confound, not as usable results.

## The confound
The old block masker set `mask_ratio` as a fraction of the SUMMER frame, but
TerraTorch's `random_masking` computes `len_keep` globally across all 768
tokens. Two consequences:

1. **Summer over-occlusion.** Nominal 20% actually masked ~33% of summer
   (84/256 patches); nominal 80% masked ~95%.
2. **Context frames were never full.** Spring/fall visibility fell with ratio
   (~25% at nominal 80%), contradicting the "full bilateral context" premise.

The random baseline used a *different* ratio definition (fraction of all
tokens), so block and random never masked matched amounts of summer.

## What this invalidated
The "crossover" — block appearing EASIER than random at 60-80%. Under corrected
masking (spring/fall 100% visible, matched summer count), block is HARDER than
random at every ratio on all four backbones, with the gap narrowing but never
flipping. Verified on 500 chips.

## Contents
* `results_{tiny,100M,300M,600M}.csv` — old block-only runs (no `random_psnr`
  column; random came from elsewhere)
* `block_summary.csv`, `sanity_check_block_vs_random.xlsx` — old aggregates
* `figures/` — old crossover-dependent figures
* `*.py.OLD` — scripts that reproduce the confound. `plot_block_results.py.OLD`
  carries a hardcoded `RANDOM_PSNR` dict; the `*_600M.py.OLD` scripts merge
  corrected block numbers with old random numbers from
  `multi_tile_generalization/outputs/per_tile/` and will silently produce
  nonsense if run.

Do not run anything in this folder.
