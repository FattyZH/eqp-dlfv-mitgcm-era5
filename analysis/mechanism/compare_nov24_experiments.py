"""Audit four Nov24 runs; compare complete runs on established time/geometry conventions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'analysis/comparison'),str(ROOT/'analysis/mechanism')]
from compare_channel_transport import model_geometry,reference_geometry,common_operator,integrate,meta
from compare_mooring_transport import iafilt,stats,sha
from diagnose_depth_amplitude import filt
from compute_ctrl_ceof import decompose
OUT=ROOT/'research/mechanism/2026-11-25_new_four';FIG=ROOT/'fig/261125_new_four'
RUNS={'VI_KPP':'261124_VI_KPP','VI_AV_KPP':'261124_VI_AV_KPP','Nr38':'261124_Nr38','Nr38_gsm':'261124_Nr38_gsm'}
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'output/261016_ctrl1';PRIOR=ROOT/'research/mechanism/2026-11-24_vector_invariant'
with np.load(ROOT/'research/mechanism/2026-11-16_depth_amplitude/source_sections.npz') as c:
 time=pd.DatetimeIndex(c['time']);arrays={'CTRL':c['model'],'GL12':c['gl12']};kk=c['kk']
with np.load(PRIOR/'CTRL_ceof.npz') as c:x=c['longitude'];z=c['depth'];oldmask=c['mask']
for key in ['VI_V3','VI_AV']:
 with np.load(PRIOR/(key+'_sources.npz')) as c:arrays[key]=c['equator']
qold=pd.read_csv(PRIOR/'native_transport.csv',index_col=0,parse_dates=True)
qcold=pd.read_csv(PRIOR/'long_common_fixed_metric_bp28.csv',index_col=0,parse_dates=True)
obsraw=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/monthly_raw_Sv.csv',index_col=0,parse_dates=True)
q={k:qold[k] for k in qold};common={};audit={};source={};partial={}
with xr.open_dataset(ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc') as ds:
 refgeom=reference_geometry(ds,Path('/public/home/zhanghang/data/GLO-MFC_001_030_coordinates.nc'),Path('/public/home/zhanghang/data/GLO-MFC_001_030_mask_bathy.nc'),2)
basecfg=f90nml.read(BASE/'data')
for key,run in RUNS.items():
 folder=ROOT/'output'/run;cfg=f90nml.read(folder/'data');nz=(folder/'RC.data').stat().st_size//4
 files={meta(p)[2]:p for p in folder.glob('diag3d.*.meta')};dates=pd.DatetimeIndex(sorted(files));ts=time.intersection(dates)
 normal='Execution ended Normally' in (folder/'STDOUT.0000').read_text(errors='replace')[-30000:]
 errors=[s for s in (folder/'STDERR.0000').read_text(errors='replace').splitlines() if '*** ERROR ***' in s]
 diff={g:{k:[basecfg.get(g,{}).get(k),cfg[g].get(k)] for k in set(basecfg.get(g,{}))|set(cfg[g]) if basecfg.get(g,{}).get(k)!=cfg[g].get(k)} for g in cfg};diff={g:v for g,v in diff.items() if v}
 identical={name:sha(folder/name)==sha(BASE/name) for name in ['data.exf','data.cal','data.pkg','data.obcs','data.ggl90','XG.data','YC.data','DYG.data'] if (folder/name).exists()}
 audit[key]=dict(experiment=run,nr=nz,months=len(dates),first_month=str(dates[0].date()),last_month=str(dates[-1].date()),normal_end=normal,errors=errors,full_analysis_available=bool(normal and len(ts)==len(time)),namelist_differences_CTRL=diff,identical_to_CTRL=identical,packages=dict(f90nml.read(folder/'data.pkg')),obcs=(folder/'data.obcs').read_text(),kpp=(folder/'data.kpp').read_text() if (folder/'data.kpp').exists() else None)
 m=model_geometry(folder,2);area=np.maximum(0,m['bottom']-np.maximum(m['top'],1900))*m['metric_per_degree'][:,None,:]*(m['north']-m['south'])[None,None,:]*m['face_weights'][:,None,None]
 op=common_operator(m,refgeom,1900);ma=np.asarray(op.sum(axis=1)).reshape(m['top'].shape);ra=np.asarray(op.sum(axis=0)).reshape(refgeom['top'].shape)
 audit[key].update(native_area_km2=float(area.sum()/1e6),common_area_km2=float(ma.sum()/1e6),coast_latitude=m['coast'])
 xa=np.fromfile(folder/'XG.data','>f4').reshape(192,720)[0];ya=np.fromfile(folder/'YC.data','>f4').reshape(192,720)[:,0]
 ii=np.flatnonzero((xa>=130)&(xa<=285));jj=np.flatnonzero(abs(ya)<=.5);zz=-np.fromfile(folder/'RC.data','>f4').astype(float)
 wet=np.fromfile(folder/'hFacW.data','>f4').reshape(nz,192,720)[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1]>0
 cache=OUT/(key+'_sources.npz')
 if cache.exists():
  with np.load(cache) as c:assert np.array_equal(c['time'],ts.values);eq=c['equator'];ch=c['channel']
 else:
  def read(t):
   p=files[t];n,fields,_=meta(p);a=np.memmap(p.with_suffix('.data'),'>f4',mode='r',shape=(n,nz,192,720));u=a[fields.index('UVEL')]
   eq=np.where(wet,np.asarray(u[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1],float),np.nan);count=np.isfinite(eq).sum(1);eq=np.divide(np.nansum(eq,axis=1),count,out=np.full(count.shape,np.nan),where=count>0)
   ch=np.stack([np.asarray(u[:,m['js'][0]:m['js'][-1]+1,f],float) for f in m['faces']]);ch[~m['wet']]=np.nan
   return eq,ch
  items=[]
  with ThreadPoolExecutor(max_workers=4) as pool:
   for n,item in enumerate(pool.map(read,ts),1):
    items.append(item)
    if n%60==0:print(key,n,'/',len(ts),flush=True)
  eq,ch=map(np.stack,zip(*items));np.savez_compressed(cache,equator=eq,channel=ch,time=ts.values,depth=zz)
 audit[key]['channel_nonfinite_wet']=int((~np.isfinite(ch[:,m['wet']])).sum());audit[key]['equator_nonfinite_wet']=int((~np.isfinite(eq[:,wet.any(1)])).sum())
 if audit[key]['channel_nonfinite_wet'] or audit[key]['equator_nonfinite_wet']:raise ValueError(f'{key}: nonfinite wet velocity')
 native=integrate(ch,area)[0];partial[key]=pd.Series(native,index=ts)
 # Exact native geometry for Q; interpolate only the equatorial structure onto CTRL depth centers.
 if audit[key]['full_analysis_available']:
  arrays[key]=interp1d(zz,eq,axis=1,bounds_error=False,fill_value=np.nan)(z)
  q[key]=pd.Series(native,index=ts);cm=integrate(ch,ma)[0]
  with xr.open_dataset(ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport/source_sections.nc') as c:ref=c.reference_u.values
  cr=integrate(ref,ra)[0];common[key]=(cm,cr)
  pd.DataFrame({'native_Sv':native,'model_common_Sv':cm,'GL12_common_Sv':cr},index=ts).to_csv(OUT/(key+'_transport.csv'))
 print(key,'normal',normal,'last',dates[-1],flush=True)
(OUT/'configuration_audit.json').write_text(json.dumps(audit,indent=2))
pd.concat(partial,axis=1).to_csv(OUT/'new_runs_available_native_Sv.csv')
# Short observation window: original iafilt, all103months, no end exclusion.
recent=pd.DataFrame({k:v.reindex(obsraw.index) for k,v in q.items()});recent['obs']=obsraw.obs
assert recent.notna().all().all()
days=pd.date_range(recent.index[0],recent.index[-1]+pd.offsets.MonthEnd(0),freq='D')
sm=pd.DataFrame(iafilt(recent.reindex(days,method='ffill').values),index=days,columns=recent.columns).resample('MS').mean()
recent.to_csv(OUT/'recent_raw_Sv.csv');sm.to_csv(OUT/'recent_iafilt_Sv.csv')
rows=[dict(dataset=k,reference='obs',**stats(sm[k],sm.obs)) for k in q]
pd.DataFrame(rows).to_csv(OUT/'recent_obs_metrics.csv',index=False)
long=[];fq={}
for k,v in q.items():
 f=filt(v.values[:,None],time.month.to_numpy())[36:-36,0];fq[k]=f
 long.append(dict(dataset=k,geometry='native',raw_mean_Sv=float(v.mean()),std_Sv=float(f.std()),correlation_CTRL=float(np.corrcoef(f,fq['CTRL'])[0,1])))
for k,(cm,cr) in common.items():
 fm=filt(cm[:,None],time.month.to_numpy())[36:-36,0];fr=filt(cr[:,None],time.month.to_numpy())[36:-36,0]
 long.append(dict(dataset=k,geometry='pairwise_common_GL12',**stats(fm,fr)))
pd.DataFrame(long).to_csv(OUT/'long_transport_metrics.csv',index=False)
# Common spatial intersection; recompute every dataset's CEOF on exactly this mask.
mask=np.logical_and.reduce([np.isfinite(a).all(0) for a in arrays.values()]);assert mask.sum()>19000
rf=-np.fromfile(BASE/'RF.data','>f4').astype(float);dz=np.maximum(0,np.minimum(rf[kk+1],3000)-rf[kk]);weights=np.broadcast_to(dz[:,None],mask.shape)
regional=[];checks={};modes={};means={};amplitudes={}
for k,a in arrays.items():
 means[k]=np.where(mask,np.mean(a,axis=0)*100,np.nan)
 f=filt(a,time.month.to_numpy());amplitudes[k]=np.where(mask,f[36:-36].std(0)*100,np.nan)
 analytic=np.full(f.shape,np.nan+1j*np.nan);analytic[:,mask]=hilbert(f[:,mask],axis=0)
 spatial,tc,frac,check=decompose(analytic[36:-36],mask,weights);checks[k]=check;modes[k]=(spatial,tc,frac)
 np.savez_compressed(OUT/(k+'_ceof.npz'),spatial_mode=spatial,time_coefficient=tc,explained_variance=frac,mask=mask,weights=weights,longitude=x,depth=z,time=time[36:-36].values)
 for lo,hi,region in [(130,150,'west'),(170,220,'central'),(220,255,'east')]:
  for top,bottom in [(0,300),(300,1000),(1000,1900),(1900,3000)]:
   w=np.maximum(0,np.minimum(rf[kk+1],bottom)-np.maximum(rf[kk],top))[:,None]*mask*((x>=lo)&(x<=hi))[None]
   regional.append(dict(dataset=k,region=region,top=top,bottom=bottom,local_std_cm_s=float(np.sqrt(np.nansum(amplitudes[k]**2*w)/w.sum())),mean_U_cm_s=float(np.nansum(means[k]*w)/w.sum())))
pd.DataFrame(regional).to_csv(OUT/'regional_amplitude.csv',index=False)
(OUT/'validation.json').write_text(json.dumps(dict(common_spatial_points=int(mask.sum()),previous_points=int(oldmask.sum()),source_months=len(time),statistics_months=len(time)-72,short_months=len(sm),nr38_section_remap='linear depth interpolation, no extrapolation; Q always native RF/hFacW',ceof_checks=checks),indent=2))
xr.Dataset({k+'_mean':(('depth','longitude'),means[k]) for k in arrays}|{k+'_std':(('depth','longitude'),amplitudes[k]) for k in arrays},coords=dict(depth=z,longitude=x)).to_netcdf(OUT/'equatorial_statistics.nc')
# Comparison figures.
fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
for k in ['obs','GL12','CTRL','Nr38','Nr38_gsm']:
 axs[0].plot(sm.index,sm[k],label=k,lw=2 if k=='obs' else 1.2);axs[1].plot(sm.index,sm[k]-sm[k].mean(),label=k)
for ax in axs:ax.legend(ncol=5);ax.set_ylabel('Transport (Sv)');ax.grid(alpha=.2)
axs[0].set_title('Original iafilt, all 103 months; native geometry');axs[1].set_title('Same smoothed series, means removed');fig.savefig(FIG/'transport_observation.png',dpi=160);plt.close(fig)
for label,fields,limit,cmap in [('mean_U',means,60,'RdBu_r'),('interannual_std',amplitudes,8,'viridis')]:
 fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained',sharex=True,sharey=True)
 for ax,k in zip(axs.flat,['GL12','CTRL','Nr38','Nr38_gsm']):
  im=ax.pcolormesh(x,z,fields[k],shading='nearest',cmap=cmap,vmin=-limit if label=='mean_U' else 0,vmax=limit);ax.set(title=k,ylim=(3000,0),xlabel='Longitude E',ylabel='Depth m');ax.axhline(300,color='gray',ls='--',lw=.6)
 fig.colorbar(im,ax=list(axs.flat),label='cm/s');fig.savefig(FIG/(label+'.png'),dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,4,figsize=(19,8),layout='constrained',sharey=True)
for j,k in enumerate(['GL12','CTRL','Nr38','Nr38_gsm']):
 spatial,tc,frac=modes[k];a=spatial[0];im=axs[0,j].pcolormesh(x,z,abs(a)*100,shading='nearest',cmap='viridis',vmin=0,vmax=8);axs[0,j].set_title(f'{k}: CEOF1 {frac[0]:.1%}')
 axs[1,j].pcolormesh(x,z,np.where(mask,-np.angle(a)*180/np.pi,np.nan),shading='nearest',cmap='twilight',vmin=-180,vmax=180)
 for ax in axs[:,j]:ax.set(ylim=(3000,0),xlabel='Longitude E',ylabel='Depth m')
fig.colorbar(im,ax=list(axs[0]),label='Amplitude cm/s');fig.savefig(FIG/'CEOF1.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(13,4),layout='constrained')
for k,v in partial.items():ax.plot(v.index,v,label=k)
ax.set(title='Available raw monthly transport: KPP series incomplete',ylabel='Sv');ax.legend();fig.savefig(FIG/'available_new_transport.png',dpi=160);plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
print(pd.DataFrame(regional).query('region=="west" and top==1900').to_string(index=False),flush=True)
print('DONE',OUT,flush=True)
