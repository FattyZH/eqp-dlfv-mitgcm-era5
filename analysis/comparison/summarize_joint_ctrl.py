"""Summarize the completed ck/A4 factorial comparison without an arbitrary score."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_western_deep import ROOT, bandpass, weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='fig/261016_ctrl_selection')
    args = p.parse_args()
    out = ROOT / args.out
    d = pd.read_csv(out / 'low_modes/regional_metrics.csv')
    q = d[(d.trim == 36) & (d.longitude == '130-150')]
    q.to_csv(out / 'western_low_mode_metrics.csv', index=False)
    d[d.depth.isin(['0-300', '300-1000', '1000-3000'])].to_csv(
        out / 'basin_low_mode_metrics.csv', index=False)
    exps = d.experiment.drop_duplicates().tolist()
    labels = ['0.10 / 2', '0.07 / 2', '0.10 / 3', '0.07 / 3',
              '0.07 / 4', '0.085 / 2', '0.085 / 3']
    if len(exps) != len(labels):
        labels = exps
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, key, title, ideal in zip(
        axes.flat, ['mean_bias_cm_s', 'r', 'std_ratio', 'rmse_cm_s'],
        ['Mean bias (cm/s)', '2–8 yr correlation', 'Std ratio (model/CMEMS)',
         'Anomaly RMSE (cm/s)'], [0, 1, 1, 0]):
        for depth in ['1000-3000', '1000-1500', '1500-2000', '2000-3000']:
            v = q[q.depth == depth].set_index('experiment').reindex(exps)
            ax.plot(np.arange(len(exps)), v[key], marker='o', label=depth + ' m')
        ax.axhline(ideal, color='gray', lw=.7, ls='--')
        ax.set(title=title, xticks=np.arange(len(exps)))
        ax.set_xticklabels(labels, rotation=25)
        ax.set_xlabel('ck / A4 (10¹⁰ m⁴/s)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / 'western_candidate_metrics.png', dpi=170)
    plt.close(fig)
    # Factorial interaction in the response itself, not differences of correlations.
    cell = ['261008_ck_0p10', '261008_ck_0p07',
            '261012_a4_3e10', '261016_ck_0p07_a4_3e10']
    fields = {e: xr.load_dataset(out / f'low_modes/{e}_reconstruction.nc').low_u
              for e in cell}
    rows = []
    for west, east in [(130, 150), (170, 220), (220, 280)]:
        for lo, hi in [(0, 300), (300, 1000), (1000, 3000),
                       (1000, 1500), (1500, 2000), (2000, 3000)]:
            s = {e: a.sel(lon=slice(west, east)).weighted(weights(a, lo, hi))
                 .mean(('depth', 'lon')) for e, a in fields.items()}
            interaction = s[cell[3]] - s[cell[1]] - s[cell[2]] + s[cell[0]]
            rows.append(dict(longitude=f'{west}-{east}', depth=f'{lo}-{hi}',
                             mean_interaction_cm_s=float(interaction.mean() * 100),
                             interannual_interaction_std_cm_s=float(
                                 bandpass(interaction, 36).std() * 100)))
    pd.DataFrame(rows).to_csv(out / 'factorial_interaction.csv', index=False)
    # Spatial errors complement errors of a region-averaged series, which can cancel.
    ref = xr.load_dataset(out / 'low_modes/CMEMS_reconstruction.nc').low_u
    cf = bandpass(ref, 36)
    rows = []
    for exp in exps:
        u = xr.load_dataset(out / f'low_modes/{exp}_reconstruction.nc').low_u
        mf = bandpass(u, 36)
        err = ((mf - cf) ** 2).mean('time')
        bias = (u - ref).mean('time')
        corr = xr.corr(mf, cf, dim='time')
        vm, vc = mf.var('time'), cf.var('time')
        for west, east in [(130, 150), (170, 220), (220, 280)]:
            for lo, hi in [(0, 300), (300, 1000), (1000, 3000), (2000, 3000)]:
                def avg(a):
                    return float(a.sel(lon=slice(west, east)).weighted(
                        weights(a, lo, hi)).mean(('depth', 'lon')))
                rows.append(dict(experiment=exp, longitude=f'{west}-{east}',
                                 depth=f'{lo}-{hi}',
                                 spatial_mean_bias_rms_cm_s=100*np.sqrt(avg(bias**2)),
                                 spatial_anomaly_rmse_cm_s=100*np.sqrt(avg(err)),
                                 spatial_mean_correlation=avg(corr),
                                 spatial_std_ratio=np.sqrt(avg(vm)/avg(vc))))
    pd.DataFrame(rows).to_csv(out / 'low_mode_spatial_metrics.csv', index=False)


if __name__ == '__main__':
    main()
