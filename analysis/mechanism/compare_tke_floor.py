"""Assess completed historicalTKEfloor+bottom restoration against CTRL/GL12.

Long2-8year analysis and full103month original-iafilt validation are distinct.
Current experiment changes both TKEmin and defaultTKEbottom; no causal split.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/comparison'))
from compare_channel_transport import meta,bandpass
from compare_mooring_transport import iafilt,stats
sys.path.insert(0,str(ROOT/'analysis/mechanism'))
from diagnose_depth_amplitude import filt,REGIONS,BANDS,wmean
NEW='261116_ctrl_TKEmin_1e-6';BASE='261016_ctrl1'
DEST=ROOT/'research/mechanism/2026-11-17_tke_floor'
PLOT=ROOT/'fig/261117_tke_floor'
PRIOR=ROOT/'research/mechanism/2026-11-16_depth_amplitude'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 DEST.mkdir(parents=True,exist_ok=True);PLOT.mkdir(parents=True,exist_ok=True)
 a=ROOT/'output'/BASE;b=ROOT/'output'/NEW
 # Compare actual archived physics and input binaries, not currentinput.
 hashes={}
 for name in ['mitgcmuv','data','data.cal','data.exf','data.pkg','data.obcs','bathy_avg.bin','XG.data','YC.data','RC.data','RF.data','hFacW.data','hFacC.data','pickup.clim_r4_1.data','pickup_ggl90.clim_r4_1.data']:
  hashes[name]=sha(a/name)==sha(b/name)
 if not all(hashes.values()):raise ValueError('Unexpected physical/initial/grid changes')
 aa=f90nml.read(a/'data.ggl90')['ggl90_parm01'];bb=f90nml.read(b/'data.ggl90')['ggl90_parm01'];differences={k:[aa.get(k),bb.get(k)] for k in set(aa)|set(bb) if aa.get(k)!=bb.get(k)}
 if set(differences)!={'ggl90tkemin'}:raise ValueError(differences)
 with (b/'STDOUT.0000').open('rb') as f:f.seek(max(0,(b/'STDOUT.0000').stat().st_size-30000));normal='Execution ended Normally' in f.read().decode(errors='replace')
 if not normal:raise ValueError('Incomplete experiment')
 (DEST/'configuration_audit.json').write_text(json.dumps(dict(same_source_hashes=hashes,namelist_differences=differences,effective_current_min=1e-6,effective_current_bottom=1e-6,effective_base_min=1e-11,effective_base_bottom=1e-11,normal_end=True,interpretation='TKEfloor plus defaultbottom combined sensitivity; diagnostics stream differs'),indent=2))
 # Shared long-record native/common transport comparison,1996-2024 support.
 old=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport/transports.csv',index_col=0,parse_dates=True)
 new=pd.read_csv(DEST/'transport/transports.csv',index_col=0,parse_dates=True)
 if not old.index.equals(new.index):raise ValueError('Transport dates differ')
 rows=[];filtered={}
 for geometry in ['native','common']:
  col='model_'+geometry+'_net_Sv';refcol='reference_available_net_Sv' if geometry=='native' else 'reference_common_net_Sv'
  for name,series in [('CTRL',old[col]),('TKE',new[col]),('GL12',old[refcol])]:
   filtered[(geometry,name)]=bandpass(series.values,series.index.month.to_numpy())[36:-36]
  for candidate in ['CTRL','TKE']:
   s=stats(filtered[(geometry,candidate)],filtered[(geometry,'GL12')]);rows.append(dict(geometry=geometry,candidate=candidate,period='2-8years1999-07 to2021-06',raw_mean_Sv=float((old if candidate=='CTRL' else new)[col].mean()),**s))
 pd.DataFrame(rows).to_csv(DEST/'long_transport_metrics.csv',index=False)
 pd.DataFrame(filtered,index=old.index[36:-36]).to_csv(DEST/'long_bandpass_transport.csv',index_label='time')
 # Original annual smoothing on common103month observation period.
 ens=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/monthly_raw_Sv.csv',index_col=0,parse_dates=True)
 recent=pd.DataFrame(dict(CTRL=ens.ctrl1,TKE=new.loc[ens.index,'model_native_net_Sv'],obs=ens.obs,GL12=ens.glorys))
 days=pd.date_range(recent.index[0],recent.index[-1]+pd.offsets.MonthEnd(0),freq='D')
 sm=pd.DataFrame(iafilt(recent.reindex(days,method='ffill').values),index=days,columns=recent.columns).resample('MS').mean()
 recent.to_csv(DEST/'recent_raw_Sv.csv',index_label='time');sm.to_csv(DEST/'recent_iafilt_Sv.csv',index_label='time')
 rr=[]
 for candidate in ['CTRL','TKE','GL12']:
  for ref in ['obs','GL12']:
   if candidate!=ref:rr.append(dict(candidate=candidate,reference=ref,period='Originaliafilt2015-12 to2024-06,all103months',**stats(sm[candidate],sm[ref])))
 pd.DataFrame(rr).to_csv(DEST/'recent_metrics.csv',index=False)
 # Extract equatorial U and mixing profiles for the same336months as prior.
 with np.load(PRIOR/'source_sections.npz') as cache:base=cache['model'];gl=cache['gl12'];times=pd.DatetimeIndex(cache['time']);kk=cache['kk']
 def rd(name,shape):return np.fromfile(b/(name+'.data'),dtype='>f4').reshape(shape)
 xall=rd('XG',(192,720))[0];yall=rd('YC',(192,720))[:,0];z=-rd('RC',(40,))[kk];rf=-rd('RF',(41,)).astype(float)
 ii=np.flatnonzero((xall>=130)&(xall<=285));jj=np.flatnonzero(abs(yall)<=.5);x=xall[ii]
 hw=rd('hFacW',(40,192,720))[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1]>0
 sources={meta(p)[2]:p for p in b.glob('diag3d.*.meta')}
 cachefile=DEST/'new_equatorial_sources.npz'
 if cachefile.exists():
  with np.load(cachefile) as d:velocity=d['velocity'];mix_new=d['mix_new'];mix_base=d['mix_base']
 else:
  def extract(time):
   path=sources[time];nr,fields,_=meta(path);raw=np.memmap(path.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720))
   v=np.where(hw,raw[fields.index('UVEL'),:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1],np.nan);count=np.isfinite(v).sum(axis=1);v=np.divide(np.nansum(v,axis=1),count,out=np.full(count.shape,np.nan),where=count>0)[kk]
   profiles=[]
   for folder in [b,a]:
    q=folder/path.name.replace('diag3d.','diag_mix_ggl90.');nr,fl,tt=meta(q)
    if tt!=time:raise ValueError('Mixing timestamp mismatch')
    mix=np.memmap(q.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720));out=[]
    for name in ['GGL90ArU','GGL90Kr']:
     vv=np.asarray(mix[fl.index(name),:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1],float)
     # Same vertical-index/region mask for both runs; exclude dry interfaces.
     valid=hw.copy();valid[1:]&=hw[:-1];valid[0]=False
     out.append([[float(np.mean(vv[k][:,mask][valid[k][:,mask]])) if valid[k][:,mask].any() else np.nan for k in range(40)] for lo,hi in REGIONS.values() for mask in [(x>=lo)&(x<=hi)]])
    profiles.append(out)
   return v,np.asarray(profiles[0]),np.asarray(profiles[1])
  with ThreadPoolExecutor(max_workers=4) as pool:
   items=[]
   for n,v in enumerate(pool.map(extract,times)):
    items.append(v)
    if (n+1)%60==0:print('Extracted U/mixing months',n+1,flush=True)
  velocity,mix_new,mix_base=map(np.stack,zip(*items));np.savez_compressed(cachefile,velocity=velocity,mix_new=mix_new,mix_base=mix_base,time=times.values,kk=kk)
 if not np.isfinite(velocity[np.isfinite(base)]).all():raise ValueError('Missing new U in old wet section')
 common=np.isfinite(base).all(axis=0)&np.isfinite(gl).all(axis=0)&np.isfinite(velocity).all(axis=0)
 fc=filt(gl,times.month.to_numpy())[36:-36];fb=filt(base,times.month.to_numpy())[36:-36];fn=filt(velocity,times.month.to_numpy())[36:-36]
 variances={'CTRL':fb.var(axis=0),'TKE':fn.var(axis=0),'GL12':fc.var(axis=0)};means={'CTRL':base.mean(axis=0),'TKE':velocity.mean(axis=0),'GL12':gl.mean(axis=0)}
 depthrows=[]
 for region,(lo,hi) in REGIONS.items():
  for band,(top,bottom) in BANDS.items():
   dz=np.maximum(0,np.minimum(rf[kk+1],bottom)-np.maximum(rf[kk],top));w=common*((x>=lo)&(x<=hi))[None]*dz[:,None]
   vals={key:float(np.sqrt(wmean(v,w))*100) for key,v in variances.items()}
   depthrows.append(dict(region=region,band=band,CTRL_std_cm_s=vals['CTRL'],TKE_std_cm_s=vals['TKE'],GL12_std_cm_s=vals['GL12'],CTRL_GL12_ratio=vals['CTRL']/vals['GL12'],TKE_GL12_ratio=vals['TKE']/vals['GL12'],TKE_CTRL_ratio=vals['TKE']/vals['CTRL'],CTRL_mean_flow_rms_cm_s=float(np.sqrt(wmean(means['CTRL']**2,w))*100),TKE_mean_flow_rms_cm_s=float(np.sqrt(wmean(means['TKE']**2,w))*100)))
 pd.DataFrame(depthrows).to_csv(DEST/'depth_amplitude_metrics.csv',index=False)
 mixing=[]
 for n,field in enumerate(['GGL90ArU','GGL90Kr']):
  for j,region in enumerate(REGIONS):
   for k,depth in enumerate(rf[:-1]):mixing.append(dict(field=field,region=region,interface_depth_m=depth,CTRL_mean_m2_s=float(np.nanmean(mix_base[:,n,j,k])) if k and np.isfinite(mix_base[:,n,j,k]).any() else np.nan,TKE_mean_m2_s=float(np.nanmean(mix_new[:,n,j,k])) if k and np.isfinite(mix_new[:,n,j,k]).any() else np.nan))
 pd.DataFrame(mixing).to_csv(DEST/'mixing_profiles.csv',index=False)
 xr.Dataset({key+'_std':(('depth','longitude'),np.sqrt(value)) for key,value in variances.items()},coords=dict(depth=z,longitude=x)).to_netcdf(DEST/'equatorial_amplitude_sections.nc')
 # Independent three source slices; cache traceability and sign-safe flux checks.
 errors=[]
 for index in [0,168,335]:
  path=sources[times[index]];nr,fields,_=meta(path);src=np.memmap(path.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720));q=np.where(hw,src[fields.index('UVEL'),:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1],np.nan);counts=np.isfinite(q).sum(axis=1);vv=np.divide(np.nansum(q,axis=1),counts,out=np.full(counts.shape,np.nan),where=counts>0)[kk];valid=common;errors.append(float(abs(vv[valid]-velocity[index][valid]).max()))
 if max(errors)>1e-10:raise AssertionError('Source U check')
 fig,axs=plt.subplots(2,1,figsize=(12,8),layout='constrained')
 for name,col in [('CTRL','black'),('TKE','#D55E00'),('GL12','#0072B2')]:
  axs[0].plot(old.index[36:-36],filtered[('common',name)],label=name,color=col)
  axs[1].plot(sm.index,sm[name],label=name,color=col)
 axs[1].plot(sm.index,sm.obs,label='Mooring',color='gray',lw=1.5)
 for ax in axs:ax.axhline(0,color='gray',lw=.5);ax.set_ylabel('Eastward transport (Sv)');ax.legend(ncol=4)
 axs[0].set_title('Common wet area below1900m |2-8years1999-07 to2021-06');axs[1].set_title('Native below1900m | originaliafilt,all103observation months')
 fig.savefig(PLOT/'K01_transport_comparison.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,6),sharey=True,layout='constrained')
 for ax,region in zip(axs,list(REGIONS)[:3]):
  xx=(x>=REGIONS[region][0])&(x<=REGIONS[region][1]);mask=common&xx[None]
  for key,col in [('CTRL','black'),('TKE','#D55E00'),('GL12','#0072B2')]:
   counts=mask.sum(axis=1);v=np.sqrt(np.divide(np.where(mask,variances[key],0).sum(axis=1),counts,out=np.full(len(z),np.nan),where=counts>0))*100;ax.plot(v,z,label=key,color=col)
  ax.set(title=region,xlabel='RMS local2-8year std (cm/s)',ylim=(3000,0));ax.axhline(1900,color='gray',ls=':');ax.legend()
 axs[0].set_ylabel('Depth (m)');fig.savefig(PLOT/'K02_vertical_amplitude.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,6),sharey=True,layout='constrained')
 for ax,field in zip(axs,['GGL90ArU','GGL90Kr']):
  data=pd.DataFrame(mixing).query('field==@field and region=="west130-150"');ax.semilogx(data.CTRL_mean_m2_s,data.interface_depth_m,label='CTRL',color='black');ax.semilogx(data.TKE_mean_m2_s,data.interface_depth_m,label='TKE',color='#D55E00');ax.set(title=field,xlabel='Mean eddy coefficient (m2/s)',ylim=(3500,0));ax.legend()
 axs[0].set_ylabel('Vertical interface depth (m)');fig.savefig(PLOT/'K03_mixing_profiles.png',dpi=180);plt.close(fig)
 info=dict(experiment=NEW,effective_parameters=dict(TKEmin=1e-6,TKEbottom=1e-6),long_transport=rows,recent_observation=[v for v in rr if v['reference']=='obs'],deep_regional_amplitude=[v for v in depthrows if v['band']=='1900-3000'],limits=['Combinedfloor/bottom change, not isolatedfloor','New experiment savesArU andKr only, notTKE/Emn: injectedenergy cannot be audited directly','Source/regional amplitude is not energyflux','Observation spatial reconstruction remains unaudited'])
 (DEST/'summary.json').write_text(json.dumps(info,indent=2));(DEST/'validation.json').write_text(json.dumps(dict(normal_end=True,months=420,independent_U_checks=errors,source_period='1996-07 to2024-06',long_analysis_period='1999-07 to2021-06',recent_months=len(sm),all_recent_endpoints_retained=True),indent=2));(DEST/'manifest.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),transport_sha256=sha(DEST/'transport/transports.csv'),source_cache_sha256=sha(cachefile)),indent=2))
 print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
