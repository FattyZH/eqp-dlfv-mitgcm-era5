"""Full-depth equatorial U comparison using cached experiment sections."""
import argparse
import f90nml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_western_deep import ROOT, bandpass, weights, metrics
EXPS=['261010_a4_5e9','261010_a4_10e9','261008_ck_0p10','261010_a4_40e9']
LABELS=['A4=5e9','A4=1e10','A4=2e10','A4=4e10']
BANDS=[(0,100),(100,300),(300,700),(700,1000),(1000,1500),(1500,2000),(2000,3000)]
REGIONS=[(130,150),(140,170),(170,220),(220,280)]
OUT=ROOT/'fig/a4_full_section_comparison'


def upper_core(fields,names,out):
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(13,6),sharey=True,constrained_layout=True)
    for ax,lon in zip(axes,[180,220,260]):
        for a,name in zip(fields['mean'],names):
            a=a.sel(lon=lon,method='nearest')
            b=a.sel(depth=slice(50,300));k=int(b.argmax('depth'))
            rows.append(dict(longitude=float(a.lon),experiment=name,core_u_cm_s=float(b.isel(depth=k)),core_depth_m=float(b.depth.isel(depth=k))))
            a=a.sel(depth=slice(0,500));ax.plot(a,a.depth,label=name,ls='--' if name=='CMEMS' else '-')
        ax.set(title=f'{lon}°E',xlabel='Mean U (cm/s)',ylim=(500,0));ax.grid(alpha=.2)
    axes[0].set_ylabel('Depth (m)');axes[0].legend(fontsize=8)
    fig.savefig(out/'upper_mean_profiles.png',dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'upper_eastward_core.csv',index=False)


def main():
    global EXPS,LABELS,OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp',nargs='+',default=EXPS)
    parser.add_argument('--out',default='fig/a4_full_section_comparison')
    parser.add_argument('--labels',nargs='+',default=None)
    args=parser.parse_args();EXPS=args.exp;OUT=ROOT/args.out
    LABELS=[f"A4={float(f90nml.read(ROOT/'output'/e/'data')['parm01']['visca4']):.2g}" for e in EXPS]
    if args.labels is not None:
        if len(args.labels)!=len(EXPS):raise ValueError('One label per experiment is required')
        LABELS=args.labels
    OUT.mkdir(parents=True,exist_ok=True)
    ds=xr.align(*[xr.load_dataset(ROOT/'fig'/f'{e}_cmems_comparison/sections.nc') for e in EXPS],join='inner')
    good=ds[0].cmems.notnull().all('time')
    for d in ds:good=good&d.model.notnull().all('time')
    raw=[ds[0].cmems.where(good)]+[d.model.where(good) for d in ds]
    names=['CMEMS']+LABELS
    rows=[]
    for west,east in REGIONS:
        for lo,hi in BANDS:
            def avg(a):
                a=a.sel(lon=slice(west,east))
                return a.weighted(weights(a,lo,hi)).mean(('depth','lon'))
            series=[avg(a) for a in raw]
            for trim in [24,36,48]:
                signals=[bandpass(a,trim) for a in series]
                for i,e in enumerate(EXPS,1):
                    rows.append(dict(experiment=e,label=names[i],trim=trim,longitude=f'{west}-{east}',depth=f'{lo}-{hi}',mean_cmems_cm_s=float(series[0].mean()*100),mean_model_cm_s=float(series[i].mean()*100),mean_bias_cm_s=float((series[i]-series[0]).mean()*100),**metrics(signals[i].values,signals[0].values)))
    pd.DataFrame(rows).to_csv(OUT/'regional_metrics.csv',index=False)
    filt=[bandpass(a,36) for a in raw]
    fields={'mean': [a.mean('time')*100 for a in raw], 'std':[a.std('time')*100 for a in filt]}
    xr.Dataset({f'{k}_{i}':a for k,vals in fields.items() for i,a in enumerate(vals)}).to_netcdf(OUT/'section_fields.nc')
    upper_core(fields,names,OUT)
    for lo,hi,title in [(0,1000,'upper_middle'),(1000,3000,'deep')]:
        fig,axes=plt.subplots(4,len(EXPS),figsize=(4.5*len(EXPS),13),sharex=True,sharey=True,constrained_layout=True)
        arrays=[fields['mean'][1:],[a-fields['mean'][0] for a in fields['mean'][1:]],fields['std'][1:],[xr.corr(a,filt[0],dim='time') for a in filt[1:]]]
        for r,vals in enumerate(arrays):
            vals=[a.sel(depth=slice(lo,hi)) for a in vals]
            vmax=1 if r==3 else max(float(np.nanquantile(np.abs(a),.98)) for a in vals)
            for ax,a,label in zip(axes[r],vals,LABELS):
                im=ax.pcolormesh(a.lon,a.depth,a,shading='auto',cmap='viridis' if r==2 else 'RdBu_r',vmin=0 if r==2 else -vmax,vmax=vmax)
                ax.set(title=label,ylim=(hi,lo),xlabel='Longitude (°E)');ax.axvline(142,color='k',ls='--',lw=.5)
            axes[r,0].set_ylabel('Depth (m)')
            fig.colorbar(im,ax=list(axes[r]),label=['Mean U (cm/s)','Mean bias (cm/s)','2–8 yr std (cm/s)','2–8 yr correlation'][r])
        fig.savefig(OUT/f'{title}_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,10),sharex=True,constrained_layout=True)
    for r,(lo,hi) in enumerate([(0,300),(300,1000),(1000,3000)]):
        for col,key in enumerate(['mean','std','correlation']):
            ax=axes[r,col]
            for i,name in enumerate(names):
                if key=='correlation' and i==0:continue
                a=raw[i].weighted(weights(raw[i],lo,hi)).mean('depth')
                b=filt[i].weighted(weights(filt[i],lo,hi)).mean('depth')
                ref=filt[0].weighted(weights(filt[0],lo,hi)).mean('depth')
                v=a.mean('time')*100 if key=='mean' else b.std('time')*100 if key=='std' else xr.corr(b,ref,dim='time')
                ax.plot(v.lon,v,label=name,lw=1.4,ls='--' if i==0 else '-')
            ax.set(title=f'{lo}–{hi} m: {key}',xlabel='Longitude (°E)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(OUT/'longitude_profiles.png',dpi=180);plt.close(fig)
    # Horizontal Hovmoller panels for upper and middle layers.
    fig,axes=plt.subplots(3,len(names),figsize=(4*len(names),12),sharex=True,sharey=True,constrained_layout=True)
    for r,(lo,hi) in enumerate([(0,100),(100,300),(300,1000)]):
        vals=[a.weighted(weights(a,lo,hi)).mean('depth')*100 for a in filt]
        vmax=max(float(np.nanquantile(np.abs(a),.98)) for a in vals)
        for ax,a,name in zip(axes[r],vals,names):
            im=ax.pcolormesh(a.lon,a.time,a,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='auto');ax.set(title=f'{name}\n{lo}–{hi} m',xlabel='Longitude (°E)')
        fig.colorbar(im,ax=list(axes[r]),label='2–8 yr U (cm/s)')
    fig.savefig(OUT/'upper_middle_hovmoller.png',dpi=150);plt.close(fig)
    print(OUT)

if __name__=='__main__':main()
