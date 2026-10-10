"""Compare model and CMEMS at a station; does not replace mooring validation."""
import argparse
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mitkit.io import open_mds
from assess_western_deep import ROOT, DEST, bandpass, weights, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp',nargs='+',default=['260821_ctrl','260916_ctrl_kpp'])
    p.add_argument('--lon',type=float,default=142)
    p.add_argument('--lat',type=float,default=0)
    p.add_argument('--trim',type=int,default=36)
    args=p.parse_args()
    if args.trim<=0: raise ValueError('trim must be positive')
    DEST.mkdir(parents=True,exist_ok=True)
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(13,8),sharey=True,constrained_layout=True)
    for exp in args.exp:
        cache=DEST/f'{exp}_site_{args.lon:g}E_{args.lat:g}N.nc'
        if cache.exists():
            section=xr.load_dataset(cache)
        else:
            src=open_mds(ROOT/'output'/exp,prefix='diag3d')
            lon=src.XG.values; lat=src.YC.values
            ix=int(np.searchsorted(lon,args.lon)); iy=int(np.searchsorted(lat,args.lat))
            if ix<1 or ix>=len(lon) or iy<1 or iy>=len(lat): raise ValueError('Station outside interpolation domain')
            indexers={'XG':slice(ix-1,ix+1),'YC':slice(iy-1,iy+1)}
            # Subset before interpolation and masking to bound memory usage.
            m=src.UVEL.isel(**indexers).where(src.hFacW.isel(**indexers)>0).interp(XG=args.lon,YC=args.lat)
            m=m.rename({'Z':'depth'}).assign_coords(depth=-src.Z.values).sortby('depth')
            m=m.drop_vars(['XG','YC'])
            m=m.assign_coords(time=pd.DatetimeIndex(m.time.values).to_period('M').to_timestamp())
            obs=xr.open_dataset(ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc')
            c=obs.uo.sel(longitude=slice(args.lon-.2,args.lon+.2),latitude=slice(args.lat-.2,args.lat+.2)).interp(longitude=args.lon,latitude=args.lat)
            c=c.drop_vars(['longitude','latitude'])
            dates=np.intersect1d(m.time.values,c.time.values)
            depths=m.depth.where((m.depth>=c.depth.min())&(m.depth<=c.depth.max()),drop=True)
            m=m.sel(time=dates,depth=depths).compute(scheduler='single-threaded')
            c=c.sel(time=dates).load().interp(depth=m.depth)
            good=np.isfinite(m).all('time')&np.isfinite(c).all('time')
            section=xr.Dataset({'model':m.where(good),'cmems':c.where(good)})
            section.attrs.update(longitude=args.lon,latitude=args.lat,interpolation='linear at original point, then CMEMS depth interpolated to model centers')
            section.to_netcdf(cache)
        m,c=bandpass(section.model,args.trim),bandpass(section.cmems,args.trim)
        profile=xr.Dataset({'correlation':xr.corr(m,c,dim='time'),'std_ratio':m.std('time')/c.std('time'),'mean_model_cm_s':section.model.mean('time')*100,'mean_cmems_cm_s':section.cmems.mean('time')*100})
        deep_profile=profile.sel(depth=slice(1000,3000))
        axes[0].plot(deep_profile.mean_model_cm_s,deep_profile.depth,label=exp)
        axes[1].plot(m.std('time').sel(depth=slice(1000,3000))*100,deep_profile.depth,label=exp)
        axes[2].plot(deep_profile.correlation,deep_profile.depth,label=exp)
        if exp==args.exp[0]:
            axes[0].plot(deep_profile.mean_cmems_cm_s,deep_profile.depth,'k--',label='CMEMS')
            axes[1].plot(c.std('time').sel(depth=slice(1000,3000))*100,deep_profile.depth,'k--',label='CMEMS')
        profile.to_dataframe().to_csv(DEST/f'{exp}_site_{args.lon:g}E_{args.lat:g}N_profile.csv')
        for lo,hi in [(1000,3000),(1000,1500),(1500,2000),(2000,3000)]:
            w=weights(m,lo,hi)
            a,b=m.weighted(w).mean('depth'),c.weighted(w).mean('depth')
            bias=(section.model-section.cmems).weighted(w).mean('depth').mean('time')
            rows.append(dict(experiment=exp,longitude=args.lon,latitude=args.lat,depth=f'{lo}-{hi}',mean_bias_cm_s=float(bias*100),**metrics(a.values,b.values)))
    pd.DataFrame(rows).to_csv(DEST/f'site_{args.lon:g}E_{args.lat:g}N_metrics.csv',index=False)
    axes[0].set_ylabel('Depth (m)')
    for ax,label in zip(axes,['Mean U (cm/s)','2–8 yr standard deviation (cm/s)','2–8 yr correlation']):
        ax.set(xlabel=label,ylim=(3000,1000)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); axes[2].set_xlim(-1,1)
    fig.suptitle(f'{args.lon:g}°E, {args.lat:g}°N: point comparison against CMEMS')
    fig.savefig(DEST/f'site_{args.lon:g}E_{args.lat:g}N_profiles.png',dpi=180); plt.close(fig)
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
