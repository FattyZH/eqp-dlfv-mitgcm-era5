"""Exploratory radius-scale estimates from phase branches and actual bathymetry.

No model input edits. Phase contours are candidate long-wave rays, not proven
energy paths; use WKB stretched depth integral(N dz) for scale estimates.
"""
from pathlib import Path
import sys,json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
import seawater as sw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_depth_amplitude import filt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/comparison'))
from compare_channel_transport import meta
OUT=ROOT/'research/mechanism/2026-11-20_radius_geometry'
FIG=ROOT/'fig/261120_radius_geometry'
MODEL=ROOT/'output/261016_ctrl1'
BASE=ROOT/'research/mechanism/2026-11-16_depth_amplitude'
R0=6370000.

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 def rd(n,shape):return np.fromfile(MODEL/f'{n}.data',dtype='>f4').reshape(shape).astype(float)
 xg=rd('XG',(192,720))[0];xc=rd('XC',(192,720))[0];y=rd('YC',(192,720))[:,0]
 rf=-rd('RF',(41,));zall=-rd('RC',(40,));bottom=(rd('hFacW',(40,192,720))*np.diff(rf)[:,None,None]).sum(axis=0)
 j=np.flatnonzero(abs(y)<=.5)
 with np.load(BASE/'source_sections.npz') as a:times=pd.DatetimeIndex(a['time']);kk=a['kk'];fields={'CTRL':a['model'],'GL12':a['gl12']}
 with xr.open_dataset(BASE/'section_statistics.nc') as d:x=d.longitude.values;z=d.depth.values
 dates=times[36:-36]
 # Representative full-column mean T/S upstream; mean first, then EOS80 N2.
 prof=OUT/'CTRL_upstream_mean_TS.npz'
 if prof.exists():
  with np.load(prof) as a:T=a['theta'];S=a['salinity']
 else:
  ii=np.flatnonzero((xc>=160)&(xc<=220));hc=rd('hFacC',(40,192,720))
  complete=hc[kk[-1],j[0]:j[-1]+1,ii[0]:ii[-1]+1]>=.999
  assert complete.sum()>0
  paths={meta(p)[2]:p for p in MODEL.glob('diag3d.*.meta')}
  def read(t):
   p=paths[t];nr,names,_=meta(p);a=np.memmap(p.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720))
   return np.stack([np.array(a[names.index(v),:len(kk),j[0]:j[-1]+1,ii[0]:ii[-1]+1],float)[:,complete].mean(axis=1) for v in ['THETA','SALT']])
  vals=[]
  with ThreadPoolExecutor(max_workers=4) as pool:
   for n,v in enumerate(pool.map(read,dates),1):
    vals.append(v)
    if n%66==0:print('Upstream T/S months',n,flush=True)
  T,S=np.mean(vals,axis=0)
  np.savez_compressed(prof,theta=T,salinity=S,depth=z,full_columns=int(complete.sum()),time=dates.values)
 pressure=sw.pres(z,0);insitu=sw.temp(S,T,pressure,pr=0)
 n2,_,pm=sw.bfrq(S,insitu,pressure,lat=0)
 zn=sw.dpth(pm[:,0],0);n2=n2[:,0]
 assert np.isfinite(n2).all()
 zg=np.arange(0.,3001.)
 Nmodel=np.interp(zg,zn,np.sqrt(np.maximum(n2,0)))
 ref=loadmat(ROOT/'archive/retained_2026-11-13/analysis/ceof/N.mat')
 Nref=np.interp(zg,ref['z'].ravel(),ref['n'].ravel())
 profiles={'CTRL_upstream':Nmodel,'retained_reference':Nref}
 pd.DataFrame(dict(depth_m=zg,N_CTRL_s1=Nmodel,N_reference_s1=Nref)).to_csv(OUT/'stratification.csv',index=False)
 stretched={k:cumulative_trapezoid(v,zg,initial=0) for k,v in profiles.items()}
 # Compare actual model U-face bathymetry with GL12 native bottom profiles.
 with xr.open_dataset('/public/home/zhanghang/data/GLO-MFC_001_030_mask_bathy.nc') as d:
  gh=d.deptho.sel(latitude=slice(-3,3),longitude=slice(140,175)).load()
 gx=gh.longitude.values;gy=gh.latitude.values;gv=gh.values
 topo=[]
 for key,xx,yy,hh in [('CTRL',xg,y,bottom),('GL12',gx,gy,gv)]:
  for width in [.5,2]:
   jj=abs(yy)<=width;ix=(xx>=140)&(xx<=175)
   a=hh[jj][:,ix];a=np.where(a>0,a,np.nan)
   for lon,v in zip(xx[ix],a.T):
    topo.append(dict(dataset=key,latitude_half_width=width,longitude=lon,shallowest_m=float(np.nanmin(v)),median_m=float(np.nanmedian(v)),deepest_m=float(np.nanmax(v))))
 table=pd.DataFrame(topo);table.to_csv(OUT/'bathymetry_profiles.csv',index=False)
 bank=(xg>=154)&(xg<=160)
 # At each longitude, deepest path within band is the optimistic 2D clearance.
 bank_min=np.min(bottom[j][:,bank],axis=0);bank_max=np.max(bottom[j][:,bank],axis=0)
 ib=np.argmin(bank_max);bank_lon=float(xg[bank][ib]);H=float(bank_max[ib]);Hshallow=float(bank_min[ib])
 print('Bank optimistic throat',bank_lon,H,'shallow envelope',Hshallow,flush=True)
 # Harmonic fits to existing 2–8 year fields, with a constant intercept.
 t=np.arange(len(dates))/12;wavefields={}
 for key,raw in fields.items():
  f=filt(raw,times.month.to_numpy())[36:-36];valid=np.isfinite(f).all(axis=0)
  with np.load(ROOT/f'research/mechanism/2026-11-17_ctrl_gl12_ceof/{key}_ceof_RF_common.npz') as a:wavefields[(key,'CEOF1')]=a['spatial_mode'][0]
  for period in [22/7,4.5]:
   A=np.column_stack([np.ones(len(t)),np.cos(2*np.pi*t/period),np.sin(2*np.pi*t/period)])
   coeff=np.linalg.lstsq(A,f[:,valid],rcond=None)[0]
   q=np.full(valid.shape,np.nan+1j*np.nan);q[valid]=coeff[1]-1j*coeff[2]
   wavefields[(key,f'harmonic_{period:.3f}y')]=q
 # Fit phase-line segments upstream in stretched depth, then extrapolate to bank.
 branches=[]
 for (dataset,name),q in wavefields.items():
  amp=abs(q)*100
  for threshold in [.25,.5]:
   for angle in np.arange(-180,180,30):
    rotated=q*np.exp(1j*np.deg2rad(angle))
    valid=np.isfinite(q)&(amp>=threshold)&(rotated.real>0)&(z[:,None]>=900)&(z[:,None]<=2700)&(x[None]>=165)&(x[None]<=215)
    fig,ax=plt.subplots();cs=ax.contour(x,z,np.ma.masked_where(~valid,rotated.imag),levels=[0])
    segments=cs.allsegs[0];plt.close(fig)
    for segment in segments:
     if len(segment)<12:continue
     xx,zz=segment.T
     if np.ptp(xx)<20:continue
     # Weight uniformly in longitude rather than by contour vertex density.
     unique=np.unique(np.round(xx,3));zz=np.array([np.median(zz[np.isclose(xx,a,atol=.001)]) for a in unique]);xx=unique
     for strat,U in stretched.items():
      yy=np.interp(zz,zg,U);slope,intercept=np.polyfit(xx,yy,1);pred=slope*xx+intercept
      r2=1-float(np.sum((yy-pred)**2)/np.sum((yy-yy.mean())**2))
      if slope>=0 or r2<.85:continue
      Ur=slope*bank_lon+intercept
      if not U[0]<Ur<U[-1]:continue
      depth=float(np.interp(Ur,U,zg))
      for start_depth in [200,500]:
       Us=np.interp(start_depth,zg,U);source_lon=(Us-intercept)/slope
       if not max(xx)<source_lon<=285:continue
       for clearance in [0,100]:
        Ulimit=np.interp(H-clearance,zg,U)
        scale=(Ulimit-Us)/(Ur-Us)
        branches.append(dict(dataset=dataset,field=name,phase_deg=int(angle),threshold_cm_s=threshold,stratification=strat,fit_lon_min=float(xx.min()),fit_lon_max=float(xx.max()),stretched_depth_r2=r2,start_depth_m=start_depth,inferred_source_lon=source_lon,bank_lon=bank_lon,bank_depth_optimistic_m=H,extrapolated_unobstructed_depth_m=depth,clearance_m=clearance,required_scale=float(scale),required_reduction_percent=float(max(0,1-scale)*100),radius_m=float(min(1,scale)*R0)))
 pd.DataFrame(branches).to_csv(OUT/'phase_branch_scale_estimates.csv',index=False)
 # Transparent scenario table even when individual phase branches are ambiguous.
 scenarios=[]
 for strat,U in stretched.items():
  for source_depth in [200,500]:
   Us=np.interp(source_depth,zg,U)
   for incident in [2000,2200,2400,2600]:
    Ui=np.interp(incident,zg,U)
    for reduction in [.03,.05,.0556,.08,.10,.15]:
     new=float(np.interp(Us+(1-reduction)*(Ui-Us),U,zg))
     scenarios.append(dict(stratification=strat,source_depth_m=source_depth,baseline_unobstructed_depth_m=incident,reduction_percent=reduction*100,radius_m=R0*(1-reduction),new_depth_m=new,lift_m=incident-new,optimistic_bank_clearance_m=H-new))
 pd.DataFrame(scenarios).to_csv(OUT/'radius_lift_scenarios.csv',index=False)
 # Wide sections to expose actual shoals and phase-contour ambiguity.
 for name in ['CEOF1','harmonic_3.143y','harmonic_4.500y']:
  fig,axs=plt.subplots(1,2,figsize=(15,6),sharey=True,layout='constrained')
  for ax,key in zip(axs,['GL12','CTRL']):
   q=wavefields[(key,name)];amp=abs(q)*100
   im=ax.pcolormesh(x,z,amp,cmap='viridis',vmin=0,vmax=4,shading='nearest')
   for angle in [-150,-90,-30,30,90,150]:
    rotated=q*np.exp(1j*np.deg2rad(angle));a=np.ma.masked_where((amp<.3)|(rotated.real<=0),rotated.imag)
    cs=ax.contour(x,z,a,levels=[0],colors='white',linewidths=.6)
    if any(len(s)>3 for s in cs.allsegs[0]):ax.clabel(cs,fmt={0:str(angle)+'°'},fontsize=6)
   for dataset,color in [('CTRL','black'),('GL12','red')]:
    a=table.query('dataset==@dataset and latitude_half_width==0.5')
    ax.plot(a.longitude,a.deepest_m,color=color,label=dataset+' deepest bottom within ±0.5°')
    ax.fill_between(a.longitude,a.shallowest_m,a.deepest_m,color=color,alpha=.1)
   ax.axvline(bank_lon,color='orange',ls='--');ax.axhline(1900,color='gray',ls=':')
   ax.set(xlim=(140,230),ylim=(3500,500),xlabel='Longitude (°E)',title=key+' '+name);ax.set_facecolor('.8');ax.legend(fontsize=7,loc='upper right')
  axs[0].set_ylabel('Depth (m)');fig.colorbar(im,ax=list(axs),label='Spatial magnitude (cm/s); white phase contours above0.3cm/s',extend='max')
  fig.suptitle('Candidate phase paths and actual bathymetry | not a verified energy-ray reconstruction')
  fig.savefig(FIG/f'R01_{name}_bathymetry.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for key,color in [('CTRL','black'),('GL12','red')]:
  a=table.query('dataset==@key and latitude_half_width==0.5');axs[0].plot(a.longitude,a.deepest_m,label=key+' deepest',color=color);axs[0].fill_between(a.longitude,a.shallowest_m,a.deepest_m,color=color,alpha=.15)
 axs[0].set(xlim=(150,165),ylim=(5500,1000),xlabel='Longitude (°E)',ylabel='Bottom depth (m)',title='Shoal range within equatorial ±0.5°');axs[0].legend()
 a=pd.DataFrame(scenarios).query("stratification=='CTRL_upstream' and source_depth_m==500")
 for incident in [2000,2200,2400,2600]:
  b=a.query('baseline_unobstructed_depth_m==@incident');axs[1].plot(b.reduction_percent,b.new_depth_m,'o-',label=f'Unobstructed depth {incident}m')
 axs[1].axhline(H,color='black',ls='--',label=f'Optimistic shoal {H:.0f}m');axs[1].axhline(H-100,color='gray',ls=':');axs[1].set(xlabel='Radius reduction (%)',ylabel='Predicted depth at fixed longitude (m)',ylim=(2800,1200),title='Fixed-N WKB geometry only; anchor depth500m');axs[1].legend(fontsize=8)
 fig.savefig(FIG/'R02_radius_lift.png',dpi=180);plt.close(fig)
 summary=dict(bank_lon=bank_lon,bank_optimistic_depth_m=H,bank_shallow_envelope_m=Hshallow,baseline_radius_m=R0,negative_N2_count=int((n2<0).sum()),phase_branch_fits=len(branches),formula='S(z)=integral N dz; S(z_new)=S(z_source)+radius_scale*(S(z_old)-S(z_source))',caveats=['Phase contours may mix waves and are not uniquely identified incident ray centers','Linear fits are in stretched depth and extrapolated west of data segment','Fixed stratification, fixed source longitude/depth, same long-wave branch; no mean-flow or feedback changes','Shoal criterion uses deepest path within±0.5°, optimistic and not a 3D connectivity proof','Two profiles bracket structure heuristically, not confidence intervals','EOS80 from mean CTRL S/PT is a representative profile, not exact MITgcm EOS or local WKB tracking'])
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary),flush=True)
 if branches:print(pd.DataFrame(branches).query("dataset=='CTRL' and clearance_m==100").groupby(['field','stratification']).required_reduction_percent.agg(['count','min','median','max']).to_string(),flush=True)

if __name__=='__main__':main()
