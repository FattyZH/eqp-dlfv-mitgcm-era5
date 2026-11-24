"""Equatorial time-mean zonal velocity, common CTRL/GL12/VI grid."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/mechanism/2026-11-24_vector_invariant';FIG=ROOT/'fig/261124_vector_invariant'
with np.load(ROOT/'research/mechanism/2026-11-16_depth_amplitude/source_sections.npz') as d:
 arrays={'GL12':d['gl12'],'CTRL':d['model']};time=d['time'];kk=d['kk']
for k in ['VI_V3','VI_AV']:
 with np.load(OUT/(k+'_sources.npz')) as d:
  assert np.array_equal(d['time'],time)
  arrays[k]=d['equator']
with np.load(OUT/'CTRL_ceof.npz') as d:x=d['longitude'];z=d['depth'];mask=d['mask']
assert np.array_equal(mask,np.logical_and.reduce([np.isfinite(v).all(axis=0) for v in arrays.values()]))
means={k:np.where(mask,v.mean(axis=0)*100,np.nan) for k,v in arrays.items()}
rf=-np.fromfile(ROOT/'output/261016_ctrl1/RF.data',dtype='>f4').astype(float)
ze=np.minimum(np.r_[rf[kk],rf[kk[-1]+1]],3000)
xe=np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
period=f'{pd.Timestamp(time[0]):%Y-%m} to {pd.Timestamp(time[-1]):%Y-%m}'
fields={k:(('depth','longitude'),v) for k,v in means.items()}
for k in ['CTRL','VI_V3','VI_AV']:fields[k+'_minus_GL12']=(('depth','longitude'),means[k]-means['GL12'])
for k in ['VI_V3','VI_AV']:fields[k+'_minus_CTRL']=(('depth','longitude'),means[k]-means['CTRL'])
xr.Dataset(fields,coords=dict(depth=z,longitude=x),attrs=dict(units='cm/s',positive='eastward',period=period,months=len(time),latitude_average='wet-point arithmetic mean within +/-0.5 degrees; reference remapped to model grid',mean='raw monthly means equally weighted; no filtering or climatology removal',mask='same20149points as paired CEOF',depth_domain='RF clipped to3000m')).to_netcdf(OUT/'equatorial_mean_U.nc')
def plot(fields,tag,top,bottom,difference=False):
 fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained',sharex=True,sharey=True)
 visible=(z>=top)&(z<=bottom)
 vmax=float(np.nanpercentile(np.abs(np.array(list(fields.values()))[:,visible]),99.5))
 for ax,(key,a) in zip(axs.flat,fields.items()):
  im=ax.pcolormesh(xe,ze,a,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='flat')
  levels=[-10,-5,-2,-1,0,1,2,5,10] if top>=1000 else [-80,-40,-20,-10,-5,0,5,10,20,40,80]
  if difference:levels=[-20,-10,-5,0,5,10,20] if top<1000 else [-5,-2,-1,0,1,2,5]
  cs=ax.contour(x,z,a,levels=levels,colors='k',linewidths=.4,alpha=.55);ax.clabel(cs,fmt='%g',fontsize=7)
  ax.set(title=key,xlim=(130,285),ylim=(bottom,top),xlabel='Longitude (E)',ylabel='Depth (m)');ax.set_facecolor('.85')
  if top<300<bottom:ax.axhline(300,color='magenta',ls='--',lw=.7)
  if top<1900<bottom:ax.axhline(1900,color='gray',ls=':',lw=.7)
 fig.colorbar(im,ax=list(axs.flat),label='Mean zonal velocity difference (cm/s)' if difference else 'Mean zonal velocity (cm/s), eastward positive',extend='both',shrink=.9)
 fig.suptitle(f'Equatorial +/-0.5 deg | time-mean zonal velocity | {period}, {len(time)} months\nRaw monthly mean; no filtering | shared color scale within each figure')
 path=FIG/(tag+'.png');fig.savefig(path,dpi=180);plt.close(fig);print(path,flush=True)
plot(means,'M01_mean_U_full',0,3000)
plot(means,'M02_mean_U_upper',0,700)
plot(means,'M03_mean_U_deep',1000,3000)
diffs={'VI_V3 minus CTRL':means['VI_V3']-means['CTRL'],'VI_AV minus CTRL':means['VI_AV']-means['CTRL'],'CTRL minus GL12':means['CTRL']-means['GL12'],'VI_AV minus GL12':means['VI_AV']-means['GL12']}
plot(diffs,'M04_mean_U_difference_full',0,3000,True)
plot(diffs,'M05_mean_U_difference_upper',0,700,True)
rows=[]
for k,a in means.items():
 valid=(x>=170)&(x<=220)
 n=np.isfinite(a[:,valid]).sum(axis=1);v=np.divide(np.nansum(a[:,valid],axis=1),n,out=np.full(len(z),np.nan),where=n>0)
 for j in range(len(z)):rows.append(dict(dataset=k,depth_m=z[j],central170_220_mean_U_cm_s=v[j]))
pd.DataFrame(rows).to_csv(OUT/'central_mean_U_profile.csv',index=False)
t=pd.DataFrame(rows)
print(t.loc[t.depth_m.between(240,350)].to_string(index=False),flush=True)
