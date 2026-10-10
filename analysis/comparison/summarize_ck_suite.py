"""Summarize a completed ck suite from aligned section and station caches."""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_western_deep import ROOT, DEST, bandpass, weights, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp',nargs='+',default=['261008_ck_0p07','261008_ck_0p10','261008_ck_0p14'])
    args=p.parse_args()
    out=ROOT/'fig'/'ck_sensitivity_suite';out.mkdir(parents=True,exist_ok=True)
    ds=xr.align(*[xr.load_dataset(ROOT/'fig'/f'{e}_cmems_comparison/sections.nc') for e in args.exp],join='inner')
    good=ds[0].cmems.notnull().all('time')
    for d in ds:
        good=good&d.model.notnull().all('time')
        if not np.allclose(d.cmems,ds[0].cmems,equal_nan=True):raise ValueError('Inconsistent CMEMS references')
    raw=[ds[0].cmems.where(good)]+[d.model.where(good) for d in ds]
    names=['CMEMS']+args.exp
    rows=[]
    for trim in [24,36,48]:
        filt=[bandpass(a,trim) for a in raw]
        for west,east in [(130,150),(140,170),(170,210),(210,270)]:
            for lo,hi in [(1000,3000),(1000,1500),(1500,2000),(2000,3000)]:
                def avg(a):
                    a=a.sel(lon=slice(west,east))
                    return a.weighted(weights(a,lo,hi)).mean(('depth','lon'))
                for i in range(1,len(raw)):
                    rows.append(dict(experiment=names[i],trim=trim,longitude=f'{west}-{east}',depth=f'{lo}-{hi}',mean_bias_cm_s=float(avg(raw[i]-raw[0]).mean()*100),**metrics(avg(filt[i]).values,avg(filt[0]).values)))
    pd.DataFrame(rows).to_csv(out/'regional_metrics.csv',index=False)
    site=xr.align(*[xr.load_dataset(DEST/f'{e}_site_142E_0N.nc') for e in args.exp],join='inner')
    rows=[]
    for trim in [24,36,48]:
        for e,d in zip(args.exp,site):
            m,c=bandpass(d.model,trim),bandpass(d.cmems,trim)
            for lo,hi in [(1000,3000),(1000,1500),(1500,2000),(2000,3000)]:
                w=weights(m,lo,hi)
                rows.append(dict(experiment=e,trim=trim,depth=f'{lo}-{hi}',mean_bias_cm_s=float((d.model-d.cmems).weighted(w).mean('depth').mean('time')*100),**metrics(m.weighted(w).mean('depth').values,c.weighted(w).mean('depth').values)))
    pd.DataFrame(rows).to_csv(out/'site_metrics.csv',index=False)
    filt=[bandpass(a,36) for a in raw]
    fig,axes=plt.subplots(3,len(names),figsize=(4*len(names),12),sharex=True,sharey=True,constrained_layout=True)
    for r,(lo,hi) in enumerate([(1000,1500),(1500,2000),(2000,3000)]):
        fields=[a.sel(lon=slice(130,180)).weighted(weights(a,lo,hi)).mean('depth')*100 for a in filt]
        vmax=max(float(np.nanquantile(np.abs(a),.98)) for a in fields)
        for ax,a,name in zip(axes[r],fields,names):
            im=ax.pcolormesh(a.lon,a.time,a,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='auto')
            ax.set_title(f'{name}\n{lo}–{hi} m');ax.axvline(142,color='k',lw=.5,ls='--')
            ax.set_xlabel('Longitude (°E)')
        fig.colorbar(im,ax=list(axes[r]),label='2–8 yr U (cm/s)')
    fig.savefig(out/'western_layer_hovmoller.png',dpi=160);plt.close(fig)
    print(out)

if __name__=='__main__':main()
