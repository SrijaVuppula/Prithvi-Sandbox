# Caution: random-masking results, OLD masking convention

These `*_results.csv` files (column `masked_psnr`) come from
`multi_tile_generalization/scripts/run_generalization.py`, which used
`patch_masking_study/masking/patch_masker.py :: build_noise_for_mask_ratio`.

They are valid for the multi-tile generalization study on their own terms.

## Do NOT merge these with block_masking_study/outputs/results_*.csv

Under this old convention:
  * `mask_ratio` = fraction of ALL 768 tokens (spring+summer+fall), NOT of the
    summer frame alone.
  * Spring/fall context frames are NOT fully visible; their visibility falls as
    mask_ratio rises (TerraTorch's random_masking computes len_keep globally).

The corrected block study (`block_masking_study/outputs/results_*.csv`) uses:
  * `mask_ratio` = fraction of the SUMMER frame only
  * spring/fall pinned at 100% visible
  * block and random matched on summer patch count

Merging the two mixes different occlusion amounts AND different context
conditions. That confound produced the retracted "crossover" finding
(block appearing easier than random at high ratios).

If you need a block-vs-random comparison, use the paired columns
(`block_psnr`, `random_psnr`, `delta_rand_minus_block`) already present in
`block_masking_study/outputs/results_*.csv`. They are measured under identical
conditions in the same run.

See `block_masking_study/outputs/old_confounded/` for the superseded artifacts.
