"""Compare cached equatorial U sections from mixing sensitivity experiments."""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_western_deep import ROOT, bandpass, weights, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default='260915_ctrl1')
    p.add_argument('--experiment',default='261008_ck_0p07')
    p.add_argument('--trim',type=int,default=36)
    args=p.parse_args()
    dest=ROOT/'fig'/f'{args.experiment}_vs_{args.baseline}'
    dest.mkdir(parents=True,exist_ok=True)
    datasets=[xr.load_dataset(ROOT/'fig'/f'{e}_cmems_comparison/sections.nc') for e in [args.baseline,args.experiment]]
    b,n=xr.align(*datasets,join='inner')
    if not np.allclose(b.cmems,n.cmems,equal_nan=True):
        raise ValueError('CMEMS references differ on common grid')
    good=np.isfinite(b.model).all('time')&np.isfinite(n.model).all('time')&np.isfinite(b.cmems).all('time')
    raw=[b.cmems.where(good),b.model.where(good),n.model.where(good)]
    filtered=[bandpass(a,args.trim) for a in raw]
    names=['CMEMS',args.baseline,args.experiment]
    rows=[]
    for west,east in [(130,150),(140,170),(170,210),(210,270)]:
        for lo,hi in [(1000,3000),(1000,1500),(1500,2000),(2000,3000)]:
            def avg(a):
                a=a.sel(lon=slice(west,east))
                return a.weighted(weights(a,lo,hi)).mean(('depth','lon'))
            ref=avg(filtered[0])
            for i in [1,2]:
                rows.append(dict(experiment=names[i],longitude=f'{west}-{east}',depth=f'{lo}-{hi}',mean_bias_cm_s=float(avg(raw[i]-raw[0]).mean()*100),**metrics(avg(filtered[i]).values,ref.values)))
    pd.DataFrame(rows).to_csv(dest/'regional_change_metrics.csv',index=False)
    mean=[a.mean('time')*100 for a in raw]
    std=[a.std('time')*100 for a in filtered]
    fields=[mean[0],mean[1],mean[2],mean[2]-mean[1],std[0],std[1],std[2],std[2]-std[1]]
    titles=['CMEMS mean','Baseline mean','ck experiment mean','Experiment − baseline mean','CMEMS 2–8 yr std','Baseline 2–8 yr std','ck experiment 2–8 yr std','Experiment − baseline std']
    fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True,sharey=True,constrained_layout=True)
    for k,(ax,a,title) in enumerate(zip(axes.flat,fields,titles)):
        a=a.sel(lon=slice(130,180),depth=slice(1000,3000))
        vmax=float(np.nanquantile(np.abs(a),.98)) or .1
        if k in [0,1,2]: vmax=max(float(np.nanquantile(np.abs(v.sel(lon=slice(130,180),depth=slice(1000,3000))),.98)) for v in mean)
        if k in [4,5,6]: vmax=max(float(np.nanquantile(v.sel(lon=slice(130,180),depth=slice(1000,3000)),.98)) for v in std)
        signed=k<4 or k==7
        im=ax.pcolormesh(a.lon,a.depth,a,cmap='RdBu_r' if signed else 'viridis',vmin=-vmax if signed else 0,vmax=vmax,shading='auto')
        ax.set(title=title,xlabel='Longitude (°E)',ylim=(3000,1000))
        ax.axvline(142,color='k',lw=.6,ls='--')
        fig.colorbar(im,ax=ax,label='cm/s')
    axes[0,0].set_ylabel('Depth (m)'); axes[1,0].set_ylabel('Depth (m)')
    fig.savefig(dest/'western_deep_changes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,constrained_layout=True)
    ts=[]
    for a,name in zip(filtered,names):
        a=a.sel(lon=slice(130,150))
        for ax,(lo,hi) in zip(axes,[(1000,1500),(1500,2000),(2000,3000)]):
            v=a.weighted(weights(a,lo,hi)).mean(('depth','lon'))*100
            ax.plot(v,v.time,label=name,lw=1)
            ax.set(title=f'130–150°E, {lo}–{hi} m',xlabel='2–8 yr U (cm/s)')
            ts.append(v.rename(f'{name}_{lo}_{hi}'))
    axes[0].legend(fontsize=8);fig.savefig(dest/'western_layer_timeseries.png',dpi=180);plt.close(fig)
    xr.merge(ts).to_netcdf(dest/'western_layer_timeseries.nc')
    print(dest)

if __name__=='__main__': main()
