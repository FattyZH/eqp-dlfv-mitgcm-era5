"""Plot full-domain and upper-ocean views of the saved paired VI CEOFs."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/mechanism/2026-11-24_vector_invariant'
FIG=ROOT/'fig/261124_vector_invariant'
KEYS=['GL12','CTRL','VI_V3','VI_AV']
data={}
for key in KEYS:
 with np.load(OUT/(key+'_ceof.npz')) as a:data[key]={n:a[n] for n in a.files}
x=data['CTRL']['longitude'];z=data['CTRL']['depth']
rf=-np.fromfile(ROOT/'output/261016_ctrl1/RF.data',dtype='>f4').astype(float)
rc=-np.fromfile(ROOT/'output/261016_ctrl1/RC.data',dtype='>f4')
kk=np.flatnonzero(np.isin(rc,z));assert np.all(np.diff(kk)==1)
ze=np.minimum(np.r_[rf[kk],rf[kk[-1]+1]],3000)
xe=np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
for mode in [0,1]:
 amps=np.array([abs(data[k]['spatial_mode'][mode])*100 for k in KEYS])
 threshold=.05*np.nanmax(amps)
 for tag,bottom in [('full',3000),('upper',700)]:
  vmax=float(np.nanpercentile(amps[:,z<=bottom],99))
  fig,axs=plt.subplots(2,4,figsize=(20,9),layout='constrained',sharex=True,sharey=True)
  for col,key in enumerate(KEYS):
   a=data[key]['spatial_mode'][mode];amp=abs(a)*100
   phase=np.where(amp>=threshold,-np.angle(a,deg=True),np.nan)
   im=axs[0,col].pcolormesh(xe,ze,amp,cmap='viridis',vmin=0,vmax=vmax,shading='flat')
   pm=axs[1,col].pcolormesh(xe,ze,phase,cmap='twilight_shifted',vmin=-180,vmax=180,shading='flat')
   axs[0,col].set_title(f'{key} | CEOF{mode+1}: {data[key]["explained_variance"][mode]*100:.2f}%')
   for row in [0,1]:
    ax=axs[row,col];ax.set(xlim=(130,285),ylim=(bottom,0),xlabel='Longitude (E)');ax.set_facecolor('.85');ax.axhline(300,color='magenta',ls='--',lw=.65)
    if bottom>1900:ax.axhline(1900,color='gray',ls=':',lw=.75)
   axs[1,col].set_title('Spatial lag phase')
  axs[0,0].set_ylabel('Depth (m)');axs[1,0].set_ylabel('Depth (m)')
  fig.colorbar(im,ax=list(axs[0]),label='Complex mode magnitude (cm/s)',extend='max',shrink=.9)
  fig.colorbar(pm,ax=list(axs[1]),label='Lag phase (degrees), shared 5% amplitude mask',shrink=.9)
  fig.suptitle(f'Independent RF-weighted CEOF{mode+1} | equatorial +/-0.5 deg | 2–8 years | July 1999–June 2021\nFull 0–3000 m decomposition; saved modes unchanged; phase aligned to CTRL spatial overlap')
  p=FIG/f'V0{mode+1}_ceof_{tag}.png';fig.savefig(p,dpi=180);plt.close(fig);print(p,flush=True)
