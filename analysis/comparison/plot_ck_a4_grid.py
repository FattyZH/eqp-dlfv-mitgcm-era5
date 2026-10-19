"""Annotated ck/A4 grids for completed experiments with common background mixing."""
import argparse
import json
import f90nml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle
from assess_western_deep import ROOT, bandpass, weights, metrics

EXPS = ['261008_ck_0p07', '261008_ck_0p10', '261008_ck_0p14',
        '261010_a4_5e9', '261010_a4_10e9', '261010_a4_40e9',
        '261012_a4_3e10', '261012_a4_6e10',
        '261016_ck_0p07_a4_3e10', '261016_ck_0p07_a4_4e10',
        '261016_ck_0p085_a4_2e10', '261016_ck_0p085_a4_3e10']
REGIONS = [(130, 150), (170, 220), (220, 280)]
BANDS = [(0, 300), (300, 1000), (1000, 3000),
         (1000, 1500), (1500, 2000), (2000, 3000)]
KEYS = ['mean_bias_cm_s', 'r', 'std_ratio', 'rmse_cm_s']
TITLES = ['Mean bias (cm/s); target 0', 'Correlation; target 1',
          'Std ratio; target 1', 'Anomaly RMSE (cm/s); target 0']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=None)
    parser.add_argument('--representation', choices=['low', 'raw'], default='low')
    parser.add_argument('--mask', choices=['full', 'complete'], default=None,
                        help='Full common wet section, or complete profiles for mode-matched comparison')
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    mask_kind = args.mask or ('complete' if args.representation == 'low' else 'full')
    if args.representation == 'low' and mask_kind != 'complete':
        parser.error('Low modes require complete profiles')
    default_out = 'parameter_grid' if args.representation == 'low' else 'raw_parameter_grid'
    out = ROOT / (args.out or f'fig/261016_ctrl_selection/{default_out}')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for exp in EXPS:
        folder = ROOT / 'output' / exp
        cfg = f90nml.read(folder / 'data')['parm01']
        ck = float(f90nml.read(folder / 'data.ggl90')['ggl90_parm01'].get('ggl90ck', .1))
        assert all(np.isclose(float(cfg[k]), 1e-5, rtol=1e-8, atol=0)
                   for k in ['viscar', 'diffkrt', 'diffkrs'])
        records.append(dict(experiment=exp, ck=ck, a4=float(cfg['visca4'])))
    config = pd.DataFrame(records)
    assert not config.duplicated(['ck', 'a4']).any()
    config.to_csv(out / 'experiment_grid.csv', index=False)
    cache = out / 'regional_metrics.csv'
    if cache.exists() and not args.rebuild:
        method = json.loads((out / 'method.json').read_text())
        if method.get('representation', 'low') != args.representation or method.get('mask', 'complete') != mask_kind:
            raise ValueError('Cached representation/mask differs; use another --out or --rebuild')
        df = pd.read_csv(cache)
    else:
        base = ROOT / 'fig/261016_ctrl_selection/low_modes'
        sources = [ROOT / 'fig' / f'{e}_cmems_comparison/sections.nc' for e in EXPS]
        if args.representation == 'low':
            basis = xr.load_dataset(base / 'basis.nc')
            reference = xr.load_dataset(base / 'CMEMS_reconstruction.nc').low_u
        else:
            reference = xr.load_dataset(sources[0]).cmems
        good = reference.notnull().all('time')
        if mask_kind == 'complete':
            good = good.all('depth')
        for source in sources:
            with xr.open_dataset(source) as d:
                assert np.array_equal(d.time, reference.time)
                assert np.array_equal(d.depth, reference.depth)
                assert np.array_equal(d.lon, reference.lon)
                valid = d.model.notnull().all('time')
                good = good & (valid.all('depth') if mask_kind == 'complete' else valid).load()
        if args.representation == 'low':
            phi = basis.velocity_mode.values
            w = basis.quadrature_weight.values
            height = float(w.sum())
        cf = bandpass(reference.where(good), 36)
        rows, spatial_rows = [], []
        for item, source in zip(records, sources):
            with xr.open_dataset(source) as d:
                u = d.model.where(good).transpose('time', 'depth', 'lon').load()
            if args.representation == 'low':
                coef = np.einsum('zm,tzx,z->tmx', phi, u.values, w) / height
                low = xr.DataArray(np.einsum('zm,tmx->tzx', phi, coef),
                                   coords=u.coords, dims=u.dims)
            else:
                low = u
            lf = bandpass(low, 36)
            ref = reference.where(good)
            bias = (low-ref).mean('time')
            err = ((lf-cf)**2).mean('time')
            corr = xr.corr(lf, cf, dim='time')
            vm, vc = lf.var('time'), cf.var('time')
            for west, east in REGIONS:
                for lo, hi in BANDS:
                    def avg(v):
                        return v.sel(lon=slice(west, east)).weighted(
                            weights(v, lo, hi)).mean(('depth', 'lon'))
                    m, c = avg(lf), avg(cf)
                    rows.append(dict(**item, longitude=f'{west}-{east}', depth=f'{lo}-{hi}',
                                     mean_bias_cm_s=float(avg(low-ref).mean()*100),
                                     **metrics(m.values, c.values)))
                    spatial_rows.append(dict(**item, longitude=f'{west}-{east}', depth=f'{lo}-{hi}',
                        spatial_mean_bias_rms_cm_s=float(np.sqrt(avg(bias**2))*100),
                        spatial_anomaly_rmse_cm_s=float(np.sqrt(avg(err))*100),
                        spatial_mean_correlation=float(avg(corr)),
                        spatial_std_ratio=float(np.sqrt(avg(vm)/avg(vc)))))
            print(item['experiment'], 'complete', flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        pd.DataFrame(spatial_rows).to_csv(out / 'spatial_metrics.csv', index=False)
        (out / 'method.json').write_text(json.dumps(dict(
            representation=args.representation, mask=mask_kind,
            retained_modes=list(range(6)) if args.representation == 'low' else None,
            common_wet_points=int(good.broadcast_like(reference.isel(time=0)).sum()),
            complete_longitude_count=int(good.all('depth').sum()) if 'depth' in good.dims else int(good.sum()),
            period='1996-07 to 2024-06', filter_years=[2, 8], endpoint_trim_months=36,
            basis='fig/261016_ctrl_selection/low_modes/basis.nc' if args.representation == 'low' else None,
            backgrounds_m2_s=1e-5, empty_cells='not run; no interpolation'), indent=2))
    cks = sorted(config.ck.unique())
    a4s = sorted(config.a4.unique())
    selected = json.loads((ROOT / 'fig/261016_ctrl_selection/selected_ctrl.json').read_text())['selected_experiment']

    def figure(queries, filename, title, headings=TITLES):
        fig, axes = plt.subplots(len(queries), 4, figsize=(19, 3.5*len(queries)+1),
                                 squeeze=False, constrained_layout=True)
        subset = pd.concat([df[(df.longitude == reg) & (df.depth == depth)]
                            for reg, depth in queries])
        for j, (key, heading) in enumerate(zip(KEYS, headings)):
            v = subset[key].to_numpy()
            if key == 'mean_bias_cm_s':
                maximum = max(float(np.abs(v).max()), .01)
                norm, cmap = TwoSlopeNorm(0, vmin=-maximum, vmax=maximum), 'RdBu_r'
            elif key == 'std_ratio':
                norm, cmap = TwoSlopeNorm(1, vmin=0, vmax=max(2, float(v.max()))), 'RdBu_r'
            elif key == 'r':
                norm, cmap = Normalize(float(v.min()), float(v.max())), 'viridis'
            else:
                norm, cmap = Normalize(float(v.min()), float(v.max())), 'YlOrRd'
            color = plt.get_cmap(cmap).copy()
            color.set_bad('#e6e6e6')
            for i, (reg, depth) in enumerate(queries):
                ax = axes[i, j]
                d = df[(df.longitude == reg) & (df.depth == depth)]
                table = d.pivot(index='ck', columns='a4', values=key).reindex(index=cks, columns=a4s)
                arr = table.to_numpy()
                im = ax.imshow(arr, norm=norm, cmap=color, aspect='auto', interpolation='nearest')
                for y, ck in enumerate(cks):
                    for x, a4 in enumerate(a4s):
                        value = arr[y, x]
                        if np.isfinite(value):
                            rgba = color(norm(value))
                            ink = 'white' if np.dot(rgba[:3], [.299, .587, .114]) < .45 else 'black'
                            ax.text(x, y, f'{value:.3f}', ha='center', va='center', color=ink, fontsize=10)
                        else:
                            ax.text(x, y, '—', ha='center', va='center', color='#888888')
                chosen = d[d.experiment == selected]
                if len(chosen):
                    ax.add_patch(Rectangle((a4s.index(float(chosen.iloc[0].a4))-.46,
                                            cks.index(float(chosen.iloc[0].ck))-.46),
                                           .92, .92, fill=False, edgecolor='#ff00b7', linewidth=2.5))
                ax.set(xticks=range(len(a4s)), yticks=range(len(cks)),
                       xticklabels=[f'{a/1e10:g}' for a in a4s],
                       yticklabels=[f'{c:g}' for c in cks],
                       xlabel='A4 (10¹⁰ m⁴/s; tested values)', ylabel='GGL90 ck',
                       title=f'{reg}°E, {depth} m\n{heading}')
                ax.set_xticks(np.arange(len(a4s)+1)-.5, minor=True)
                ax.set_yticks(np.arange(len(cks)+1)-.5, minor=True)
                ax.grid(which='minor', color='white', linewidth=.5)
                ax.tick_params(which='minor', bottom=False, left=False)
            fig.colorbar(im, ax=list(axes[:, j]), shrink=.85)
        representation = 'n=0–5' if args.representation == 'low' else f'Original U; {mask_kind} wet mask'
        fig.suptitle(title+f'\n{representation}; 2–8 yr; gray = not run; magenta outline = selected ctrl', fontsize=14)
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)

    figure([('130-150', '1000-3000')], 'western_deep_grid.png', 'Western deep zonal velocity comparison')
    figure([('130-150', b) for b in ['1000-1500', '1500-2000', '2000-3000']],
           'western_deep_layers_grid.png', 'Western deep layers')
    for depth, tag in [('1000-3000', 'basin_deep'), ('0-300', 'basin_upper'), ('300-1000', 'basin_middle')]:
        figure([(f'{a}-{b}', depth) for a, b in REGIONS], tag+'_grid.png', 'Equatorial basin comparison')
    description = '共同低模态n=0–5' if args.representation == 'low' else '原始纬向流U，不作垂直模态截断'
    lines = ['# ck × A4参数二维表', '', description+f'；湿点掩膜：{mask_kind}。2–8年，首尾各裁36个月。'
             '仅比较背景垂向系数均1e-5的12组实验；KPP及背景系数不同的实验不混入本表。'
             '灰色/空白表示未运行，不插值；A4列为离散已测值，列距不代表数值间距。', '']
    q = df[(df.longitude == '130-150') & (df.depth == '1000-3000')]
    for key, title in zip(KEYS, TITLES):
        table = q.pivot(index='ck', columns='a4', values=key).reindex(index=cks, columns=a4s)
        table.to_csv(out / f'western_deep_{key}_matrix.csv')
        lines += [f'## {title}', '', '| ck \\ A4（×1e10） | '+' | '.join(f'{x/1e10:g}' for x in a4s)+' |',
                  '|---|'+'---|'*len(a4s)]
        for ck, row in table.iterrows():
            lines.append(f'| {ck:g} | '+' | '.join(f'{v:.3f}' if pd.notna(v) else '—' for v in row)+' |')
        lines.append('')
    lines += ['相关和振幅比理想值1；偏差理想值0，RMSE越小越好。颜色不表示显著性。'
              '相关和RMSE色标按每张图实际数值范围展开以显示差异；同图同一指标共用色标，跨图应比较数值。'
              '区域平均时间序列指标不能替代逐点空间误差；本图不改变综合ctrl选择依据。', '',
              '[西部整层](western_deep_grid.png) · [西部分层](western_deep_layers_grid.png) · '
              '[盆地深层](basin_deep_grid.png) · [上层](basin_upper_grid.png) · [中层](basin_middle_grid.png)']
    (out / 'README.md').write_text('\n'.join(lines)+'\n')
    if args.representation == 'raw':
        df = pd.read_csv(out / 'spatial_metrics.csv').rename(columns={
            'spatial_mean_bias_rms_cm_s': 'mean_bias_cm_s',
            'spatial_mean_correlation': 'r', 'spatial_std_ratio': 'std_ratio',
            'spatial_anomaly_rmse_cm_s': 'rmse_cm_s'})
        headings = ['Spatial RMS mean bias (cm/s); target 0',
                    'Spatial mean correlation; target 1',
                    'Spatial RMS std ratio; target 1',
                    'Pointwise anomaly RMSE (cm/s); target 0']
        figure([('130-150', '1000-3000')], 'western_deep_spatial_grid.png',
               'Original U: spatial errors (no cancellation by regional averaging)', headings)
        figure([(f'{a}-{b}', '1000-3000') for a, b in REGIONS], 'basin_deep_spatial_grid.png',
               'Original U: basin spatial errors', headings)
        with (out / 'README.md').open('a') as stream:
            stream.write('\n另见[西部逐点空间指标](western_deep_spatial_grid.png)及'
                         '[盆地逐点空间指标](basin_deep_spatial_grid.png)。'
                         '平均态RMS误差和逐点年际RMSE先在各点计算，再按层厚与经度汇总，'
                         '不同于先区域平均后计算误差。两类评价均需保留。\n')


if __name__ == '__main__':
    main()
