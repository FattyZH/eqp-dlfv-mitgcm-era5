"""Independent source-data audit of the final control and its two references.

Writes research evidence, without recreating the cleaned fig caches.
"""
from pathlib import Path
import hashlib
import argparse
import json
import re
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
from scipy.signal import butter, sosfiltfilt
from scipy.io import loadmat
from scipy.linalg import eigh

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'analysis').is_dir() and (p/'input').is_dir())
OUT = ROOT / 'research/model_control/verification'
EVIDENCE = ROOT / 'research/model_control/evidence'
EXPS = ['261008_ck_0p10', '261008_ck_0p07', '261016_ck_0p07_a4_4e10']


def site_audit():
    """Check cached point series against direct source samples and saved metrics."""
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(EVIDENCE/'site_142E_metrics.csv')
    checks = []
    for exp in EXPS:
        folder = ROOT/'output'/exp
        point_snapshot = OUT/'point_series'/f'{exp}.nc'
        cached = xr.load_dataset(point_snapshot if point_snapshot.exists() else
                                 ROOT/f'research/deep_transport_assessment/{exp}_site_142E_0N.nc')
        x = np.fromfile(folder/'XG.data', dtype='>f4').reshape(192, 720)[0]
        y = np.fromfile(folder/'YC.data', dtype='>f4').reshape(192, 720)[:, 0]
        z = -np.fromfile(folder/'RC.data', dtype='>f4')
        ix, iy = int(np.searchsorted(x, 142)), int(np.searchsorted(y, 0))
        wx = np.array([x[ix]-142, 142-x[ix-1]])/(x[ix]-x[ix-1])
        wy = np.array([y[iy], -y[iy-1]])/(y[iy]-y[iy-1])
        wet = np.fromfile(folder/'hFacW.data', dtype='>f4').reshape(40, 192, 720)[:, iy-1:iy+1, ix-1:ix+1] > 0
        kk = np.array([np.argmin(abs(z-v)) for v in cached.depth.values])
        dates = pd.DatetimeIndex(cached.time.values)
        selected = set(dates[[0, 40, 80, 120, 160, 200, 260, 335]])
        errors = []
        for meta in sorted(folder.glob('diag3d.*.meta')):
            end = re.search(r"timeStepDate\s*=\s*\[\s*'([^']+)'", meta.read_text()).group(1)
            t = (pd.Timestamp(end)-pd.Timedelta(seconds=2629800/2)).tz_localize(None).to_period('M').to_timestamp()
            if t not in selected:
                continue
            a = np.memmap(meta.with_suffix('.data'), dtype='>f4', mode='r', shape=(6, 40, 192, 720))
            v = np.where(wet, a[0, :, iy-1:iy+1, ix-1:ix+1], np.nan)
            point = np.sum(v*wy[None, :, None]*wx[None, None, :], axis=(1, 2))[kk]
            target = cached.model.sel(time=t).values
            assert np.array_equal(np.isfinite(point), np.isfinite(target))
            errors.append(float(np.nanmax(abs(point-target))))
            del a
        assert len(errors) == 8 and max(errors) < 1e-6
        def filt(a):
            a = a.copy()
            for m in range(1, 13):
                take = dates.month == m
                a[take] -= np.mean(a[take], axis=0)
            return sosfiltfilt(butter(4, [1/8, 1/2], fs=12, btype='bandpass', output='sos'), a, axis=0)[36:-36]
        zz = cached.depth.values
        ee = np.r_[0, (zz[:-1]+zz[1:])/2, zz[-1]]
        metric_errors = []
        for lo, hi in [(1000, 3000), (1000, 1500), (1500, 2000), (2000, 3000)]:
            w = np.maximum(0, np.minimum(ee[1:], hi)-np.maximum(ee[:-1], lo))
            def avg(a):
                return np.nansum(a*w[None], axis=1)/np.sum(np.isfinite(a)*w[None], axis=1)
            m, c = cached.model.values, cached.cmems.values
            a, b = filt(avg(m)), filt(avg(c))
            values = dict(mean_bias_cm_s=float(np.mean(avg(m-c))*100), r=float(np.corrcoef(a, b)[0, 1]),
                          std_ratio=float(np.std(a)/np.std(b)), rmse_cm_s=float(np.sqrt(np.mean((a-b)**2))*100))
            row = old[(old.experiment == exp) & (old.depth == f'{lo}-{hi}')].iloc[0]
            metric_errors.extend(abs(value-float(row[key])) for key, value in values.items())
        assert max(metric_errors) < 2e-5
        checks.append(dict(experiment=exp, source_sample_months=8,
                           max_source_point_difference_m_s=max(errors),
                           max_saved_metric_difference=max(metric_errors)))
    report = dict(status='passed', checks=checks,
                  scope='Point metrics for the three audited experiments recomputed from retained point series; source model interpolation checked at 8 months per experiment.')
    (OUT/'site_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-source-checkpoint', action='store_true')
    parser.add_argument('--validate-saved-results', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.validate_saved_results:
        checkpoint_path = Path('/tmp/eqp_final_ctrl_audit_sources.npz')
        assert (OUT/'recomputed_regional_metrics.csv').stat().st_mtime >= checkpoint_path.stat().st_mtime
        assert (OUT/'recomputed_spatial_metrics.csv').stat().st_mtime >= checkpoint_path.stat().st_mtime
        with np.load(checkpoint_path) as saved:
            common = np.isfinite(saved['cmems']).all(axis=0)
            for exp in EXPS:
                common &= np.isfinite(saved[exp]).all(axis=0)
            kk = saved['kk']
        complete = common.all(axis=0)
        assert common.sum() == 20149 and complete.sum() == 527
        configs = json.loads(Path('/tmp/eqp_final_ctrl_audit_configs.json').read_text())
        assert all(a['hashes'] == configs[0]['hashes'] for a in configs)
        assert all(a['normal_end'] and a['monthly_files'] == 420 for a in configs)
        p = ROOT/'output'/EXPS[-1]
        z = -np.fromfile(p/'RC.data', dtype='>f4')[kk]
        rf = -np.fromfile(p/'RF.data', dtype='>f4')
        legacy_edges = np.r_[0, (z[:-1]+z[1:])/2, z[-1]]
        physical_edges = rf[np.r_[kk, kk[-1]+1]].astype(float)
        basis_error = json.loads((OUT/'coordinate_precision_check.json').read_text())['native_arithmetic_basis_max_difference']
        assert basis_error < 1e-9
        finalize_audit(configs, common, complete, basis_error, legacy_edges, physical_edges, True)
        return
    p = ROOT / 'output' / EXPS[-1]
    read = lambda name, shape: np.fromfile(p / f'{name}.data', dtype='>f4').reshape(shape)
    x = read('XG', (192, 720))[0]
    y = read('YC', (192, 720))[:, 0]
    z = -read('RC', (40,))
    rf = -read('RF', (41,))
    j = np.flatnonzero(abs(y) <= .5)
    i = np.flatnonzero((x >= 130) & (x <= 285))
    if args.resume_source_checkpoint:
        checkpoint_path = Path('/tmp/eqp_final_ctrl_audit_sources.npz')
        assert checkpoint_path.exists()
        newest_source = max([float((ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc').stat().st_mtime)] +
                            [f.stat().st_mtime for e in EXPS for f in (ROOT/'output'/e).glob('diag3d.*.data')])
        assert checkpoint_path.stat().st_mtime >= newest_source
        with np.load(checkpoint_path) as saved:
            kk = saved['kk']
            months = pd.DatetimeIndex(saved['months'])
            c = xr.DataArray(saved['cmems'], dims=('time', 'depth', 'lon'),
                             coords={'time': months, 'depth': z[kk], 'lon': x[i]})
            fields = {e: saved[e] for e in EXPS}
        configs = json.loads(Path('/tmp/eqp_final_ctrl_audit_configs.json').read_text())
        print('Resumed the freshly extracted source arrays; source files not newer than checkpoint', flush=True)
    else:
        with xr.open_dataset(ROOT / 'data/cmems/gl12_uvt_9607-2406_fd.nc') as obs:
            kk = np.flatnonzero((z >= float(obs.depth.min())) & (z <= float(obs.depth.max())))
            c = obs.uo.sel(latitude=slice(-.5, .5)).mean('latitude').load()
            c = c.rename({'longitude': 'lon'}).interp(depth=z[kk], lon=x[i])
            months = pd.DatetimeIndex(c.time.values).to_period('M').to_timestamp()
            c = c.assign_coords(time=months).transpose('time', 'depth', 'lon')
        assert np.array_equal(months, pd.date_range('1996-07-01', '2024-06-01', freq='MS'))
        wet = read('hFacW', (40, 192, 720))[:, j[0]:j[-1]+1, i[0]:i[-1]+1] > 0
        fields = {}
        configs = []
        for exp in EXPS:
            folder = ROOT / 'output' / exp
            cfg = f90nml.read(folder / 'data')['parm01']
            mix = f90nml.read(folder / 'data.ggl90')['ggl90_parm01']
            hashes = {}
            for name in ['mitgcmuv', 'pickup.clim_r4_1.data', 'pickup_ggl90.clim_r4_1.data',
                         'data.exf', 'data.obcs', 'data.pkg', 'data.cal',
                         'XG.data', 'YC.data', 'RC.data', 'RF.data', 'hFacW.data']:
                hashes[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            configs.append(dict(experiment=exp, ck=float(mix.get('ggl90ck', .1)),
                                a4=float(cfg['visca4']), hashes=hashes,
                                normal_end='Execution ended Normally' in (folder/'STDOUT.0000').read_text(),
                                monthly_files=len(list(folder.glob('diag3d.*.meta')))))
            values, dates = [], []
            for meta in sorted(folder.glob('diag3d.*.meta')):
                txt = meta.read_text()
                end = re.search(r"timeStepDate\s*=\s*\[\s*'([^']+)'", txt).group(1)
                t = (pd.Timestamp(end)-pd.Timedelta(seconds=2629800/2)).tz_localize(None).to_period('M').to_timestamp()
                if t not in months:
                    continue
                assert "'UVEL    '" in txt and 'float32' in txt
                a = np.memmap(meta.with_suffix('.data'), dtype='>f4', mode='r', shape=(6, 40, 192, 720))
                v = np.where(wet, a[0, :, j[0]:j[-1]+1, i[0]:i[-1]+1], np.nan)
                count = np.isfinite(v).sum(axis=1)
                avg = np.divide(np.nansum(v, axis=1), count,
                                out=np.full(count.shape, np.nan), where=count > 0)
                values.append(avg[kk]); dates.append(t)
                del a
            assert np.array_equal(dates, months)
            fields[exp] = np.array(values)
            print(exp, 'read 336 source months', flush=True)
    assert all(a['hashes'] == configs[0]['hashes'] for a in configs)
    assert all(a['normal_end'] and a['monthly_files'] == 420 for a in configs)
    common = np.isfinite(c.values).all(axis=0)
    for a in fields.values():
        common &= np.isfinite(a).all(axis=0)
    complete = common.all(axis=0)
    # Preserve the independent source extraction outside fig for recovery during auditing.
    if not args.resume_source_checkpoint:
        np.savez('/tmp/eqp_final_ctrl_audit_sources.npz', cmems=c.values,
                 months=months.values, kk=kk, **{e: a for e, a in fields.items()})
        Path('/tmp/eqp_final_ctrl_audit_configs.json').write_text(json.dumps(configs))
    assert common.sum() == 20149 and complete.sum() == 527, (common.sum(), complete.sum())
    c = c.values
    zz = z[kk].astype(float)
    # Match the historical native float32 grid arithmetic before promoting to float64.
    native_z = z[kk]
    midpoints = (native_z[:-1]+native_z[1:])/2
    legacy_edges = np.r_[0, midpoints, native_z[-1]]
    assert np.all(np.diff(kk) == 1)
    physical_edges = rf[np.r_[kk, kk[-1]+1]].astype(float)
    # Rebuild the documented reference modes, rather than trusting the cached basis.
    frozen_nmat = ROOT/'research/model_control/method/N.mat'
    nmat = loadmat(frozen_nmat if frozen_nmat.exists() else ROOT/'analysis/ceof/N.mat')
    height = float(nmat['z'].max())
    edges = np.r_[0, midpoints, height]
    w = np.diff(edges)
    n2 = np.interp(edges[1:-1], nmat['z'].ravel(), nmat['n'].ravel())**2
    operator = np.zeros((len(zz), len(zz)))
    for k, g in enumerate(1/(n2*np.diff(native_z))):
        operator[k, k] += g; operator[k+1, k+1] += g
        operator[k, k+1] -= g; operator[k+1, k] -= g
    _, modes = eigh(operator, np.diag(w))
    modes *= np.sqrt(height)
    modes *= np.where(modes[0] < 0, -1, 1)
    basis = xr.load_dataset(EVIDENCE/'vertical_mode_basis.nc')
    basis_error = float(np.max(abs(basis.velocity_mode.values-modes[:, :6])))
    assert basis_error < 1e-9, basis_error

    def low(a):
        a = np.where(complete[None, None, :], a, np.nan)
        coef = np.einsum('zm,tzx,z->tmx', modes[:, :6], a, w)/height
        return np.einsum('zm,tmx->tzx', modes[:, :6], coef)

    sos = butter(4, [1/8, 1/2], fs=12, btype='bandpass', output='sos')
    calendar_month = months.month.to_numpy()

    def filtered(a):
        anomaly = a.copy()
        for m in range(1, 13):
            take = calendar_month == m
            anomaly[take] -= np.mean(a[take], axis=0)
        return sosfiltfilt(sos, anomaly, axis=0)[36:-36]

    def avg(a, weight):
        rows, cols = np.nonzero(weight > 0)
        rs, cs = slice(rows.min(), rows.max()+1), slice(cols.min(), cols.max()+1)
        a = a[..., rs, cs]
        weight = weight[rs, cs]
        den = np.sum(np.isfinite(a)*weight, axis=(-2, -1))
        return np.nansum(a*weight, axis=(-2, -1))/den

    rows, spatial, series = [], [], []
    for representation in ['raw', 'low']:
        mask = common if representation == 'raw' else np.broadcast_to(complete, common.shape)
        ref = np.where(mask[None], c if representation == 'raw' else low(c), np.nan)
        cf = filtered(ref)
        for exp, raw in fields.items():
            a = np.where(mask[None], raw if representation == 'raw' else low(raw), np.nan)
            af = filtered(a)
            with np.errstate(invalid='ignore', divide='ignore'):
                mb = np.mean(a-ref, axis=0)
                err = np.mean((af-cf)**2, axis=0)
                vm, vo = np.var(af, axis=0), np.var(cf, axis=0)
                corr = np.mean((af-np.mean(af, 0))*(cf-np.mean(cf, 0)), 0)/np.sqrt(vm*vo)
            for rule, ee in [('legacy_midpoint', legacy_edges), ('physical_RF', physical_edges)]:
                for west, east in [(130, 150), (170, 220), (220, 280)]:
                    for lo, hi in [(0, 300), (300, 1000), (1000, 3000),
                                   (1000, 1500), (1500, 2000), (2000, 3000)]:
                        dz = np.maximum(0, np.minimum(ee[1:], hi)-np.maximum(ee[:-1], lo))
                        wt = dz[:, None]*((x[i] >= west) & (x[i] <= east))[None, :]
                        m, o = avg(af, wt), avg(cf, wt)
                        mse = np.mean((m-o)**2)
                        item = dict(experiment=exp, representation=representation, weighting=rule,
                                    longitude=f'{west}-{east}', depth=f'{lo}-{hi}')
                        rows.append(dict(**item, mean_bias_cm_s=float(np.mean(avg(a-ref, wt))*100),
                                         r=float(np.corrcoef(m, o)[0, 1]),
                                         std_ratio=float(np.std(m)/np.std(o)),
                                         rmse_cm_s=float(np.sqrt(mse)*100)))
                        spatial.append(dict(**item,
                            spatial_mean_bias_rms_cm_s=float(np.sqrt(avg(mb**2, wt))*100),
                            spatial_anomaly_rmse_cm_s=float(np.sqrt(avg(err, wt))*100),
                            spatial_mean_correlation=float(avg(corr, wt)),
                            spatial_std_ratio=float(np.sqrt(avg(vm, wt)/avg(vo, wt)))))
                        if west == 130 and lo == 1000 and hi == 3000:
                            series.extend(dict(**item, time=str(t.date()), model_cm_s=float(mm*100),
                                               cmems_cm_s=float(oo*100))
                                          for t, mm, oo in zip(months[36:-36], m, o))
            print(representation, exp, 'regional/spatial checks computed', flush=True)
    result = pd.DataFrame(rows); field = pd.DataFrame(spatial)
    result.to_csv(OUT/'recomputed_regional_metrics.csv', index=False)
    field.to_csv(OUT/'recomputed_spatial_metrics.csv', index=False)
    pd.DataFrame(series).to_csv(OUT/'western_deep_series.csv', index=False)
    finalize_audit(configs, common, complete, basis_error, legacy_edges, physical_edges, args.resume_source_checkpoint)


def finalize_audit(configs, common, complete, basis_error, legacy_edges, physical_edges, resumed):
    result = pd.read_csv(OUT/'recomputed_regional_metrics.csv')
    field = pd.read_csv(OUT/'recomputed_spatial_metrics.csv')
    discrepancies = []
    for representation in ['raw', 'low']:
        old = pd.read_csv(EVIDENCE/f'{representation}_regional_metrics.csv')
        new = result[(result.representation == representation) & (result.weighting == 'legacy_midpoint')]
        merged = new.merge(old, on=['experiment', 'longitude', 'depth'], suffixes=('_new', '_old'))
        assert len(merged) == 54
        for key in ['mean_bias_cm_s', 'r', 'std_ratio', 'rmse_cm_s']:
            delta = float(abs(merged[key+'_new']-merged[key+'_old']).max())
            tolerance = 1e-4 if key.endswith('cm_s') else 5e-6
            assert delta < tolerance, (representation, key, delta)
            discrepancies.append(dict(representation=representation, metric=key, max_difference=delta))
        old = pd.read_csv(EVIDENCE/f'{representation}_spatial_metrics.csv')
        merged = field[(field.representation == representation) &
                       (field.weighting == 'legacy_midpoint')].merge(
                           old, on=['experiment', 'longitude', 'depth'], suffixes=('_new', '_old'))
        assert len(merged) > 0
        for key in ['spatial_mean_bias_rms_cm_s', 'spatial_anomaly_rmse_cm_s',
                    'spatial_mean_correlation', 'spatial_std_ratio']:
            delta = float(abs(merged[key+'_new']-merged[key+'_old']).max())
            tolerance = 1e-4 if key.endswith('cm_s') else 5e-6
            assert delta < tolerance, (representation, key, delta)
            discrepancies.append(dict(representation=representation, metric=key, max_difference=delta))
    report = dict(status='passed', date='2026-10-19', source_months=336,
                  resumed_from_source_checkpoint=resumed,
                  common_wet_points=int(common.sum()), complete_profiles=int(complete.sum()),
                  basis_max_difference=basis_error, configurations=configs,
                  differences=discrepancies, nominal_band_bottom_m=3000,
                  legacy_bottom_edge_m=float(legacy_edges[-1]), physical_last_interface_m=float(physical_edges[-1]),
                  comparison_tolerances={'velocity_cm_s': 1e-4, 'dimensionless': 5e-6},
                  notes=['Native float32 coordinate arithmetic and reduction order introduce sub-rounding differences.',
                         'Physical RF weights are an additional sensitivity audit, not replacements for legacy figure labels.',
                         'Equal-longitude, layer-thickness-weighted velocity proxies, not cross-section transport.'])
    (OUT/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    site_audit()


if __name__ == '__main__':
    main()
