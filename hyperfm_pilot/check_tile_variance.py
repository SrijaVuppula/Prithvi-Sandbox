import pandas as pd

df = pd.read_csv('spectral_ratio_sweep_results.csv')

# per-tile median PSNR at 20% scattered - see if a few tiles are driving the swing
sub = df[(df['ratio'] == 0.2) & (df['mask_type'] == 'scattered')]
per_tile = sub.groupby('tile')['psnr'].agg(['median', 'mean', 'std', 'count']).sort_values('median')
print(per_tile.to_string())
print(f"\nSpread across tiles: min={per_tile['median'].min():.2f}, max={per_tile['median'].max():.2f}, range={per_tile['median'].max()-per_tile['median'].min():.2f} dB")
