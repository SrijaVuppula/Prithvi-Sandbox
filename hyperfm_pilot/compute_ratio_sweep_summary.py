import pandas as pd

df_50 = pd.read_csv('spectral_ratio_sweep_results.csv')
df_10 = pd.read_csv('spectral_ratio_sweep_results_10trials_backup.csv')

def summarize(df, label):
    print(f"\n=== {label} ===")
    summary = df.groupby(['ratio', 'mask_type']).agg(
        psnr_median=('psnr', 'median'),
        psnr_mean=('psnr', 'mean'),
        mae_median=('mae', 'median'),
        mae_mean=('mae', 'mean'),
        n=('psnr', 'count')
    ).reset_index()
    print(summary.to_string(index=False))

    print(f"\n--- Gap (contiguous - scattered), median PSNR, {label} ---")
    for ratio in sorted(df['ratio'].unique()):
        cont = df[(df['ratio'] == ratio) & (df['mask_type'] == 'contiguous')]['psnr'].median()
        scat = df[(df['ratio'] == ratio) & (df['mask_type'] == 'scattered')]['psnr'].median()
        print(f"ratio={ratio:.1f}: contiguous={cont:.2f} dB, scattered={scat:.2f} dB, gap={cont-scat:+.2f} dB")

summarize(df_10, "10 TRIALS (backup)")
summarize(df_50, "50 TRIALS (new)")
