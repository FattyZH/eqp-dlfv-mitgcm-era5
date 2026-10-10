"""Reproducible CMEMS/model comparison on their common equatorial section."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from mitkit.io import open_mds
from mitkit.paths import project_root


def monthly(da):
    return da.assign_coords(time=pd.DatetimeIndex(da.time.values).to_period('M').to_timestamp())


def filtered(da):
    anomaly = da.groupby('time.month') - da.groupby('time.month').mean('time')
    x = anomaly.values
    valid = np.isfinite(x).all(axis=0)
    out = np.full(x.shape, np.nan)
    # Same filter, climatology period and endpoint handling for both products.
    out[:, valid] = sosfiltfilt(butter(4, [1/8, 1/2], fs=12, btype='bandpass', output='sos'), x[:, valid], axis=0)
    return xr.DataArray(out, coords=da.coords, dims=da.dims)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp', default='260821_ctrl')
    p.add_argument('--rebuild-cache', action='store_true', help='Reload source data after model output changes')
    p.add_argument('--skip-deep-modes', action='store_true', help='Skip deep CEOF and Hovmoller figures')
    p.add_argument('--trim', type=int, default=36, help='Discard months at each filter endpoint')
    args = p.parse_args()
    root = project_root()
    dest = root / 'fig' / f'{args.exp}_cmems_comparison'
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / 'sections.nc'
    if cache.exists() and not args.rebuild_cache:
        sections = xr.load_dataset(cache)
    else:
        ds = open_mds(root/'output'/args.exp, prefix='diag3d')
        u = ds.UVEL.where(ds.hFacW > 0).sel(YC=slice(-.5,.5), XG=slice(130,285))
        u = u.rename({'Z':'depth','YC':'lat','XG':'lon'})
        u = monthly(u.assign_coords(depth=-u.depth).sortby('depth'))
        obs = xr.open_dataset(root/'data/cmems/gl12_uvt_9607-2406_fd.nc')
        c = monthly(obs.uo.rename({'latitude':'lat','longitude':'lon'}))
        c = c.sel(lat=slice(-.5,.5))
        dates = np.intersect1d(u.time.values, c.time.values)
        depths = u.depth.where((u.depth >= c.depth.min()) & (u.depth <= c.depth.max()), drop=True)
        u = u.sel(time=dates, depth=depths).mean('lat').compute(scheduler='single-threaded')
        c = c.sel(time=dates).mean('lat').load().interp(depth=u.depth, lon=u.lon)
        valid = np.isfinite(u).all('time') & np.isfinite(c).all('time')
        sections = xr.Dataset({'model':u.where(valid),'cmems':c.where(valid)})
        sections.to_netcdf(cache)
    m, c = sections.model, sections.cmems
    dates = pd.DatetimeIndex(m.time.values)
    if not dates.is_unique or not dates.equals(pd.date_range(dates[0], dates[-1], freq='MS')):
        raise ValueError('Comparison requires unique, consecutive monthly data; missing months must be resolved first')
    if args.trim < 0:
        raise ValueError('--trim must be nonnegative')
    if args.trim*2 >= m.sizes['time']:
        raise ValueError('Endpoint trim leaves no analysis interval')
    mf, cf = filtered(m), filtered(c)
    sl = slice(args.trim, -args.trim if args.trim else None)
    mf, cf = mf.isel(time=sl), cf.isel(time=sl)
    metrics = xr.Dataset({
        'mean_model':m.mean('time'), 'mean_cmems':c.mean('time'),
        'mean_bias':(m-c).mean('time'),
        'std_model':mf.std('time'), 'std_cmems':cf.std('time'),
        'correlation':xr.corr(mf,cf,dim='time'),
        'rmse':np.sqrt(((mf-cf)**2).mean('time')),
    })
    metrics['std_ratio'] = metrics.std_model / metrics.std_cmems
    metrics.attrs.update(filter='4th order zero-phase Butterworth, 2-8 years; monthly climatology removed', endpoint_trim_months=args.trim, experiment=args.exp)
    metrics.to_netcdf(dest/'metrics.nc')
    xr.Dataset({'model':mf,'cmems':cf}).to_netcdf(dest/'interannual_sections.nc')
    fig, axs = plt.subplots(3,2,figsize=(14,11),sharex=True,sharey=True,constrained_layout=True)
    panels = [('mean_cmems','CMEMS mean',100,'RdBu_r',-.6*100,.6*100),('mean_bias','Model minus CMEMS mean',100,'RdBu_r',-30,30),('std_cmems','CMEMS 2–8 yr standard deviation',100,'YlOrRd',0,15),('std_model','Model 2–8 yr standard deviation',100,'YlOrRd',0,15),('correlation','2–8 yr correlation',1,'RdBu_r',-1,1),('std_ratio','Model / CMEMS standard deviation',1,'RdBu_r',0,2)]
    for ax,(var,title,scale,cmap,vmin,vmax) in zip(axs.flat,panels):
        im=ax.pcolormesh(m.lon,m.depth,metrics[var]*scale,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
        ax.set_title(title); ax.set_ylim(float(m.depth.max()),0)
        ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Depth (m)')
        fig.colorbar(im,ax=ax,label='cm/s' if scale==100 else '')
    fig.savefig(dest/'section_metrics.png',dpi=180); plt.close(fig)
    # Physical cell-width weights on the common model grid, clipped to band bounds.
    z=m.depth.values; ze=np.r_[0,(z[:-1]+z[1:])/2,z[-1]]
    bands=[(0,200),(200,1000),(1000,3000)]
    regions=[(140,170),(170,220),(220,280)]
    rows=[]
    fig,axs=plt.subplots(3,3,figsize=(16,10),sharex=True,constrained_layout=True)
    spectral=[]
    for i,(lo,hi) in enumerate(bands):
        weights=xr.DataArray(np.maximum(0,np.minimum(ze[1:],hi)-np.maximum(ze[:-1],lo)),dims='depth',coords={'depth':m.depth})
        for j,(west,east) in enumerate(regions):
            def avg(a):
                return a.sel(lon=slice(west,east)).weighted(weights).mean(('depth','lon'))
            a,b=avg(mf),avg(cf)
            ax=axs[i,j]; ax.plot(a.time,a*100,label='Model'); ax.plot(b.time,b*100,label='CMEMS'); ax.axhline(0,color='k',lw=.4)
            ax.set_title(f'{west}–{east}°E, {lo}–{hi} m'); ax.set_ylabel('cm/s')
            if i==0 and j==0: ax.legend()
            r=float(xr.corr(a,b,dim='time'))
            rows.append(dict(depth=f'{lo}-{hi}',longitude=f'{west}-{east}',correlation=r,model_std_cm_s=float(a.std()*100),cmems_std_cm_s=float(b.std()*100),rmse_cm_s=float(np.sqrt(((a-b)**2).mean())*100)))
            # Unfiltered deseasonalized series for independent spectral evidence.
            for label,da in [('model',m),('cmems',c)]:
                s=avg(da); s=s.groupby('time.month')-s.groupby('time.month').mean('time')
                f,power=welch(s.values,fs=12,nperseg=min(168,len(s)),detrend='linear')
                spectral.append((i,j,label,f,power))
    fig.savefig(dest/'regional_timeseries.png',dpi=180); plt.close(fig)
    pd.DataFrame(rows).to_csv(dest/'regional_metrics.csv',index=False)
    fig,axs=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for i,j,label,f,power in spectral:
        keep=f>0; axs[i,j].plot(1/f[keep],power[keep]*1e4,label=label)
        axs[i,j].set(xscale='log',xlim=(.5,15),xlabel='Period (years)',ylabel='PSD ((cm/s)² / cpy)')
        axs[i,j].axvspan(2,8,alpha=.08,color='gray')
        axs[i,j].set_title(f'{regions[j][0]}–{regions[j][1]}°E, {bands[i][0]}–{bands[i][1]} m')
    axs[0,0].legend(); fig.savefig(dest/'regional_spectra.png',dpi=180); plt.close(fig)
    info=dict(experiment=args.exp,common_start=str(m.time.values[0])[:10],common_end=str(m.time.values[-1])[:10],filtered_start=str(mf.time.values[0])[:10],filtered_end=str(mf.time.values[-1])[:10],months=m.sizes['time'],valid_cells=int(np.isfinite(m.isel(time=0)).sum()),regional_metrics=rows)
    (dest/'summary.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
    if not args.skip_deep_modes:
        from deep_modes import analyze
        analyze(args.exp)

if __name__=='__main__':
    main()
