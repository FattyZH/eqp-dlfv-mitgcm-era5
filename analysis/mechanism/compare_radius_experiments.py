"""Radius sensitivity: audited geometry, matched CEOFs and transport validation."""
from pathlib import Path
import sys,json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/comparison'))
from compare_channel_transport import model_geometry,meta,integrate
from compare_mooring_transport import iafilt,stats,sha
from compute_ctrl_ceof import decompose
from diagnose_depth_amplitude import filt
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extended',action='store_true',help='Include 261122_R95 with its actual 6000000m radius');args=parser.parse_args()
PRIOR=ROOT/'research/mechanism/2026-11-22_radius'
EXPERIMENTS=[('R99',.99,'261120_R99'),('R98',.98,'261120_R98'),('R97',.97,'261120_R97')]
if args.extended:EXPERIMENTS.append(('R95',6000000/6370000,'261122_R95'))
KEYS=['GL12','CTRL']+[v[0] for v in EXPERIMENTS]
COLORS={'obs':'black','GL12':'#0072B2','CTRL':'#666666','R99':'#009E73','R98':'#E69F00','R97':'#9467bd','R95':'#D55E00'}
OUT=ROOT/'research/mechanism'/('2026-11-23_radius_extended' if args.extended else '2026-11-22_radius');FIG=ROOT/'fig'/('261123_radius_extended' if args.extended else '261122_radius')
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
b=ROOT/'output/261016_ctrl1'
with np.load(ROOT/'research/mechanism/2026-11-16_depth_amplitude/source_sections.npz') as d:
 arrays={'CTRL':d['model'],'GL12':d['gl12']};time=pd.DatetimeIndex(d['time']);kk=d['kk']
with xr.open_dataset(ROOT/'research/mechanism/2026-11-16_depth_amplitude/section_statistics.nc') as d:x=d.longitude.values;z=d.depth.values
old=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport/transports.csv',index_col=0,parse_dates=True)
with xr.open_dataset(ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport/source_sections.nc') as d:cu=d.model_u.values
m=model_geometry(b,2)
area=np.maximum(0,m['bottom']-np.maximum(m['top'],1900))*m['metric_per_degree'][:,None,:]*(m['north']-m['south'])[None,None,:]*m['face_weights'][:,None,None]
q={'CTRL':old.model_native_net_Sv.values,'GL12':old.reference_available_net_Sv.values};qc={'CTRL':old.model_common_net_Sv.values,'GL12':old.reference_common_net_Sv.values}
# Recover linear common-area weights from the same audited geometries.
from compare_channel_transport import reference_geometry,common_operator
with xr.open_dataset(ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc') as d:r=reference_geometry(d,Path('/public/home/zhanghang/data/GLO-MFC_001_030_coordinates.nc'),Path('/public/home/zhanghang/data/GLO-MFC_001_030_mask_bathy.nc'),2)
op=common_operator(m,r,1900);ma=np.asarray(op.sum(axis=1)).reshape(m['top'].shape)
assert np.max(abs(integrate(cu,area)[0]-q['CTRL']))<1e-10
xa=np.fromfile(b/'XG.data',dtype='>f4').reshape(192,720)[0];ya=np.fromfile(b/'YC.data',dtype='>f4').reshape(192,720)[:,0]
ii=np.flatnonzero((xa>=130)&(xa<=285));jj=np.flatnonzero(abs(ya)<=.5)
hw=np.fromfile(b/'hFacW.data',dtype='>f4').reshape(40,192,720)[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1]>0
audit={}
for name,scale,experiment in EXPERIMENTS:
 folder=ROOT/'output'/experiment;cfg=f90nml.read(folder/'data');base=f90nml.read(b/'data')
 differences={g:{k:[base.get(g,{}).get(k),cfg[g].get(k)] for k in set(base.get(g,{}))|set(cfg[g]) if base.get(g,{}).get(k)!=cfg[g].get(k)} for g in cfg};differences={g:v for g,v in differences.items() if v}
 assert set(differences)=={'parm04'} and set(differences['parm04'])=={'rsphere'},differences
 hashes={n:sha(folder/n)==sha(b/n) for n in ['mitgcmuv','data.exf','data.ggl90','data.pkg','data.cal','data.obcs','bathy_avg.bin','pickup.clim_r4_1.data','pickup_ggl90.clim_r4_1.data','XG.data','YC.data','RF.data','hFacW.data']}
 assert all(hashes.values()),hashes
 assert 'Execution ended Normally' in (folder/'STDOUT.0000').read_text(errors='replace')[-30000:]
 metric=np.fromfile(folder/'DYG.data',dtype='>f4')/np.fromfile(b/'DYG.data',dtype='>f4');assert np.max(abs(metric-scale))<1e-6
 sources={meta(p)[2]:p for p in folder.glob('diag3d.*.meta')};assert all(t in sources for t in time)
 dest=OUT/(name+'_sources.npz')
 if args.extended and not dest.exists() and (PRIOR/dest.name).exists():dest.symlink_to(PRIOR/dest.name)
 if dest.exists():
  with np.load(dest) as d:eq=d['equator'];ch=d['channel']
 else:
  def read(t):
   p=sources[t];nr,fields,_=meta(p);raw=np.memmap(p.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720));u=raw[fields.index('UVEL')]
   a=np.where(hw,np.asarray(u[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1],float),np.nan);count=np.isfinite(a).sum(axis=1)
   eq=np.divide(np.nansum(a,axis=1),count,out=np.full(count.shape,np.nan),where=count>0)[kk]
   ch=np.stack([np.asarray(u[:,m['js'][0]:m['js'][-1]+1,f],float) for f in m['faces']]);ch[~m['wet']]=np.nan
   return eq,ch
  with ThreadPoolExecutor(max_workers=4) as pool:
   items=[]
   for n,item in enumerate(pool.map(read,time),1):
    items.append(item)
    if n%60==0:print(name,'extracted',n,flush=True)
  eq,ch=map(np.stack,zip(*items));np.savez_compressed(dest,equator=eq,channel=ch,time=time.values)
 arrays[name]=eq;q[name]=integrate(ch,area*scale)[0];qc[name]=integrate(ch,ma)[0]
 audit[name]=dict(scale=scale,namelist_differences=differences,identical_hashes=hashes,normal_end=True,monthly_diagnostics=len(sources),metric_scale_max_error=float(np.max(abs(metric-scale))))
(OUT/'configuration_audit.json').write_text(json.dumps(audit,indent=2))
mask=np.logical_and.reduce([np.isfinite(a).all(axis=0) for a in arrays.values()]);assert mask.sum()==20149
rf=-np.fromfile(b/'RF.data',dtype='>f4').astype(float);dz=np.maximum(0,np.minimum(rf[kk+1],3000)-rf[kk]);weights=np.broadcast_to(dz[:,None],mask.shape)
results={};regional=[];modes=[];checks={};filtered={}
for key,a in arrays.items():
 f=filt(a,time.month.to_numpy());filtered[key]=f[36:-36];analytic=np.full(f.shape,np.nan+1j*np.nan);analytic[:,mask]=hilbert(f[:,mask],axis=0)
 cached=PRIOR/(key+'_ceof.npz')
 if args.extended and cached.exists():
  with np.load(cached) as c:
   assert np.array_equal(mask,c['mask']) and np.array_equal(weights,c['weights']) and np.array_equal(time[36:-36].values,c['time'])
   sm=c['spatial_mode'].copy();tc=c['time_coefficient'].copy();frac=c['explained_variance'].copy()
  check=json.loads((PRIOR/'ceof_validation.json').read_text())[key]
 else:sm,tc,frac,check=decompose(analytic[36:-36],mask,weights)
 if key!='CTRL':
  for i in range(4):
   rot=np.exp(-1j*np.angle(np.sum(np.conj(results['CTRL']['sm'][i,mask])*sm[i,mask]*weights[mask])));sm[i]*=rot;tc[:,i]*=np.conj(rot)
 results[key]=dict(sm=sm,tc=tc,frac=frac);checks[key]=check
 np.savez_compressed(OUT/(key+'_ceof.npz'),spatial_mode=sm,time_coefficient=tc,explained_variance=frac,mask=mask,weights=weights,longitude=x,depth=z,time=time[36:-36].values)
 for region,lo,hi in [('west',130,150),('central',170,220),('east',220,255)]:
  for top,bot in [(1000,3000),(1900,3000)]:
   w=np.maximum(0,np.minimum(rf[kk+1],bot)-np.maximum(rf[kk],top))[:,None]*mask*((x>=lo)&(x<=hi))[None]
   avg=lambda v:float(np.nansum(v*w)/w.sum())
   local=np.sqrt(avg(f[36:-36].var(axis=0)))*100
   regional.append(dict(dataset=key,region=region,top=top,bottom=bot,local_std_cm_s=local))
   for i in [0,1]:
    a0=results['CTRL']['sm'][i];a1=sm[i];wg=w[mask];v=a0[mask];u=a1[mask]
    cong=abs(np.sum(np.conj(v)*u*wg))/np.sqrt(np.sum(abs(v)**2*wg)*np.sum(abs(u)**2*wg))
    err=np.sqrt(avg(abs(a1-a0)**2))/np.sqrt(avg(abs(a0)**2))
    modes.append(dict(dataset=key,mode=i+1,region=region,top=top,variance_fraction=frac[i],mode_rms_cm_s=np.sqrt(avg(abs(sm[i])**2))*100,congruence_to_CTRL=float(cong),relative_complex_change_CTRL=float(err)))
pd.DataFrame(regional).to_csv(OUT/'regional_amplitude.csv',index=False);pd.DataFrame(modes).to_csv(OUT/'ceof_metrics.csv',index=False)
(OUT/'ceof_validation.json').write_text(json.dumps(checks,indent=2))
# Common Q uses baseline metric for every model and reference: isolates velocity response.
long=[];fq={k:filt(v[:,None],time.month.to_numpy())[36:-36,0] for k,v in qc.items()}
for k in q:
 long.append(dict(dataset=k,**stats(fq[k],fq['GL12'])))
pd.DataFrame(long).to_csv(OUT/'long_common_fixed_metric_metrics.csv',index=False)
ens=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/monthly_raw_Sv.csv',index_col=0,parse_dates=True)
recent=pd.DataFrame({k:pd.Series(v,index=time).loc[ens.index] for k,v in q.items()});recent['obs']=ens.obs
raw=recent.copy();days=pd.date_range(recent.index[0],recent.index[-1]+pd.offsets.MonthEnd(0),freq='D');sm=pd.DataFrame(iafilt(recent.reindex(days,method='ffill').values),index=days,columns=recent.columns).resample('MS').mean()
raw.to_csv(OUT/'recent_raw_Sv.csv');sm.to_csv(OUT/'recent_iafilt_Sv.csv');pd.DataFrame(q,index=time).to_csv(OUT/'native_transport.csv');pd.DataFrame(fq,index=time[36:-36]).to_csv(OUT/'long_common_fixed_metric_bp28.csv')
rr=[dict(dataset=k,reference=ref,**stats(sm[k],sm[ref])) for k in q for ref in ['obs','GL12'] if k!=ref];pd.DataFrame(rr).to_csv(OUT/'recent_metrics.csv',index=False)
for mode in [0,1]:
 keys=KEYS;vmax=np.nanpercentile([abs(results[k]['sm'][mode][z>=1000])*100 for k in keys],99)
 fig,axs=plt.subplots(2,len(keys),figsize=(4.6*len(keys),8),layout='constrained',sharey=True)
 threshold=.05*max(np.nanmax(abs(results[k]['sm'][mode])) for k in keys)
 for col,k in enumerate(keys):
  a=results[k]['sm'][mode];im=axs[0,col].pcolormesh(x,z,abs(a)*100,vmin=0,vmax=vmax,cmap='viridis',shading='nearest');phase=np.where(abs(a)>=threshold,-np.angle(a,deg=True),np.nan);pm=axs[1,col].pcolormesh(x,z,phase,vmin=-180,vmax=180,cmap='twilight_shifted',shading='nearest')
  axs[0,col].set_title(f'{k} | {results[k]["frac"][mode]*100:.2f}%')
  for row in [0,1]:axs[row,col].set(xlim=(130,220),ylim=(3000,1000),xlabel='Longitude E');axs[row,col].axvline(156.375,color='red',ls=':',lw=.8);axs[row,col].axhline(1900,color='gray',ls=':',lw=.7);axs[row,col].set_facecolor('.85')
 axs[0,0].set_ylabel('Depth (m)');axs[1,0].set_ylabel('Depth (m)');fig.colorbar(im,ax=list(axs[0]),label='Mode magnitude cm/s');fig.colorbar(pm,ax=list(axs[1]),label='Lag phase degrees');fig.suptitle(f'CEOF{mode+1}: RF weights, full 0–3000m decomposition; deep zoom; phase aligned to CTRL spatial overlap');fig.savefig(FIG/f'R0{mode+1}_ceof_deep.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(14,11),layout='constrained')
for k in ['obs']+KEYS:
 axs[0].plot(sm.index,sm[k],label=k,color=COLORS[k]);axs[1].plot(sm.index,sm[k]-sm[k].mean(),label=k,color=COLORS[k])
for k,v in fq.items():axs[2].plot(time[36:-36],v,label=k,color=COLORS[k])
for ax in axs:ax.legend(ncol=6);ax.axhline(0,color='gray',lw=.5);ax.set_ylabel('Transport Sv')
axs[0].set_title('Native geometry, original iafilt, all 103 months');axs[1].set_title('Same series, means removed');axs[2].set_title('Long 2–8yr, identical common area and CTRL metric (velocity effect)');fig.savefig(FIG/'R03_transport.png',dpi=180);plt.close(fig)
print(pd.DataFrame(rr).query('reference=="obs"').to_string(index=False),flush=True);print(pd.DataFrame(long).to_string(index=False),flush=True);print(pd.DataFrame(regional).query('region=="west"').to_string(index=False),flush=True);print(pd.DataFrame(modes).query('region=="west" and top==1900').to_string(index=False),flush=True)

# Additional GL12 spatial comparisons and fixed-metric sensitivity.
from pathlib import Path
import numpy as np,pandas as pd,json
root=ROOT;out=OUT
d={k:dict(np.load(out/(k+'_ceof.npz'))) for k in KEYS};g=d['GL12'];x=g['longitude'];z=g['depth'];mask=g['mask'];w0=g['weights'];rows=[]
for k,a in d.items():
 for region,lo,hi,top in [('full',130,285,0),('west_deep',130,150,1900),('upstream_deep',150,170,1900)]:
  rf=-np.fromfile(root/'output/261016_ctrl1/RF.data',dtype='>f4').astype(float); kk=np.flatnonzero(np.isin(-np.fromfile(root/'output/261016_ctrl1/RC.data',dtype='>f4'),z)); dz=np.maximum(0,np.minimum(rf[kk+1],3000)-np.maximum(rf[kk],top)); w=dz[:,None]*mask*((x>=lo)&(x<=hi))[None];sel=w>0;ww=w[sel]
  for i in range(4):
   v=g['spatial_mode'][i,sel];u=a['spatial_mode'][i,sel];overlap=np.sum(np.conj(v)*u*ww)/np.sqrt(np.sum(abs(v)**2*ww)*np.sum(abs(u)**2*ww));rot=np.exp(-1j*np.angle(overlap));err=np.sqrt(np.sum(abs(u*rot-v)**2*ww)/np.sum(abs(v)**2*ww))
   t=g['time_coefficient'][:,i];s=a['time_coefficient'][:,i];tc=abs(np.vdot(t,s))/np.sqrt(np.vdot(t,t).real*np.vdot(s,s).real)
   rows.append(dict(dataset=k,region=region,mode=i+1,spatial_congruence_GL12=float(abs(overlap)),relative_complex_error_GL12=float(err),complex_TC_correlation_GL12=float(tc)))
pd.DataFrame(rows).to_csv(out/'ceof_GL12_comparison.csv',index=False)
print(pd.DataFrame(rows).query('mode<=2 and region=="west_deep"').to_string(index=False))
sys_path=root/'analysis/comparison'
import sys
sys.path.insert(0,str(sys_path))
from compare_channel_transport import bandpass
from compare_mooring_transport import iafilt,stats
q=pd.read_csv(out/'native_transport.csv',index_col=0,parse_dates=True);recent=pd.read_csv(out/'recent_raw_Sv.csv',index_col=0,parse_dates=True)
fixed=recent.copy()
for k,s,_ in EXPERIMENTS:fixed[k]/=s
idx=pd.date_range(fixed.index[0],fixed.index[-1]+pd.offsets.MonthEnd(0),freq='D');smooth=pd.DataFrame(iafilt(fixed.reindex(idx,method='ffill').values),index=idx,columns=fixed.columns).resample('MS').mean()
pd.DataFrame([dict(dataset=k,**stats(smooth[k],smooth.obs)) for k in KEYS]).to_csv(out/'recent_fixed_metric_obs_metrics.csv',index=False)
rows=[]
for k in q:
 f=bandpass(q[k].values,q.index.month.to_numpy())[36:-36];rows.append(dict(dataset=k,mean_Sv=q[k].mean(),interannual_std_Sv=f.std(),correlation_CTRL=np.corrcoef(f,bandpass(q.CTRL.values,q.index.month.to_numpy())[36:-36])[0,1]))
pd.DataFrame(rows).to_csv(out/'long_native_metrics.csv',index=False)
# Independent reproduction check of the established CTRL/GL12 baseline.
validation={}
for k in ['CTRL','GL12']:
 with np.load(root/'research/mechanism/2026-11-17_ctrl_gl12_ceof'/f'{k}_ceof_RF_common.npz') as old:
  validation[k+'_fraction_max_error']=float(np.max(abs(old['explained_variance']-d[k]['explained_variance'])))
  assert validation[k+'_fraction_max_error']<1e-10
  for i in range(4):
   a=old['spatial_mode'][i,mask];b=d[k]['spatial_mode'][i,mask];rot=np.exp(-1j*np.angle(np.vdot(a,b)));validation[k+f'_mode{i+1}_phase_invariant_max_error']=float(np.max(abs(a-b*rot)))
   assert validation[k+f'_mode{i+1}_phase_invariant_max_error']<1e-10
(out/'baseline_reproduction.json').write_text(json.dumps(validation,indent=2))
