"""Equatorial sections for GL12, CTRL and historical bandpass-wind run."""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/mechanism/2026-11-17_historical_bandpass'
FIG=ROOT/'fig/261117_historical_bandpass'

def main():
    with xr.open_dataset(SOURCE/'equatorial_std.nc') as ds:
        ds=ds.load()
    x,z=ds.longitude.values,ds.depth.values
    valid=np.logical_and.reduce([np.isfinite(ds[k].values) for k in ['GL12','CTRL','BP']])
    for limit,tag in [(3000,'full'),(700,'upper'),(3000,'deep')]:
        fig,axs=plt.subplots(3,3,figsize=(17,11),sharex=True,sharey=True,layout='constrained')
        maximum=3 if tag=='deep' else 10
        for col,key in enumerate(['GL12','CTRL','BP']):
            std=np.where(valid,ds[key].values*100,np.nan)
            ratio=ds[key].values/ds.GL12.values
            r=ds[key+'_correlation_GL12'].values
            a=axs[0,col].pcolormesh(x,z,std,cmap='viridis',vmin=0,vmax=maximum,shading='nearest')
            axs[0,col].set_title({'GL12':'CMEMS / GL12','CTRL':'Current CTRL','BP':'Historical BP24–96'}[key])
            b=axs[1,col].pcolormesh(x,z,ratio,cmap='BrBG_r',norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2),shading='nearest')
            c=axs[2,col].pcolormesh(x,z,r,cmap='RdBu_r',vmin=-1,vmax=1,shading='nearest')
            if key!='GL12':
                cs=axs[2,col].contour(x,z,ratio,levels=[.5,1,1.5],colors='black',linewidths=.6)
                axs[2,col].clabel(cs,fmt={.5:'A=0.5',1:'A=1',1.5:'A=1.5'},fontsize=7)
        for row,label in enumerate(['Amplitude','Amplitude ratio to GL12','Correlation with GL12']):
            axs[row,0].set_ylabel(label+'\nDepth (m)')
        fig.colorbar(a,ax=list(axs[0]),label='2–8 year U std (cm/s)',extend='max',shrink=.8)
        fig.colorbar(b,ax=list(axs[1]),label='A = std / GL12 std',extend='max',shrink=.8)
        fig.colorbar(c,ax=list(axs[2]),label='Zero-lag Pearson r; contours: A',shrink=.8)
        for ax in axs.flat:
            ax.set_facecolor('.85')
            ax.set(xlim=(130,285),ylim=(3000,1000) if tag=='deep' else (limit,0))
            if tag!='deep':ax.axhline(300,color='magenta',ls='--',lw=.8)
            if tag!='upper':ax.axhline(1900,color='gray',ls=':',lw=.8)
            ax.axvline(142,color='gray',ls=':',lw=.7)
        for ax in axs[-1]:ax.set_xlabel('Longitude (°E)')
        fig.suptitle('Equatorial ±0.5° wet-point mean U | 2–8 years | July 1999–June 2021\nCommon wet mask; gray = unavailable | historical BP also changes A4 and ck')
        fig.savefig(FIG/f'B03_equatorial_sections_{tag}.png',dpi=180)
        plt.close(fig)
    print('Saved full, upper-700m and deep-1000–3000m equatorial sections')

if __name__=='__main__':main()
