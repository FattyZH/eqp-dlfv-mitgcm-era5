"""Compare NAS historical bandpass-wind run with current CTRL and GL12.

Read-only remote extraction, resumable local month caches; grid identity must
have been audited separately. This is not a one-factor wind experiment.
"""
from pathlib import Path
import sys,json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/comparison'))
from compare_channel_transport import model_geometry,reference_geometry,common_operator,integrate,meta
from diagnose_depth_amplitude import filt,REGIONS,BANDS
OUT=ROOT/'research/mechanism/2026-11-17_historical_bandpass'
FIG=ROOT/'fig/261117_historical_bandpass'
OLD=ROOT/'mnt/output/260825_wind_bp24-96'
BASE=ROOT/'research/mechanism/2026-11-16_depth_amplitude'

def main():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 cache=OUT/'months';cache.mkdir(exist_ok=True)
 validation=json.loads((OUT/'grid_validation.json').read_text())
 assert all(v['bp']==v['ctrl'] for v in validation.values())
 with np.load(BASE/'source_sections.npz') as a:
  ctrl=a['model'];gl=a['gl12'];time=pd.DatetimeIndex(a['time']);kk=a['kk']
 m=model_geometry(ROOT/'output/261016_ctrl1',2)
 def rd(n,shape):return np.fromfile(ROOT/'output/261016_ctrl1'/f'{n}.data',dtype='>f4').reshape(shape)
 xall=rd('XG',(192,720))[0];y=rd('YC',(192,720))[:,0];rf=-rd('RF',(41,)).astype(float)
 ii=np.flatnonzero((xall>=130)&(xall<=285));jj=np.flatnonzero(abs(y)<=.5)
 hw=rd('hFacW',(40,192,720))[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1]>0
 src=[]
 for p in sorted(OLD.glob('diag3d.*.meta')):
  nr,fields,t=meta(p)
  if t in time:src.append((p,nr,fields,t))
 assert pd.DatetimeIndex([v[3] for v in src]).equals(time)
 def read(item):
  p,nr,fields,t=item;dest=cache/f'{t:%Y-%m}.npz'
  if dest.exists():
   with np.load(dest) as a:return a['equator'],a['channel']
  raw=np.memmap(p.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720))
  u=raw[fields.index('UVEL')]
  a=np.where(hw,np.array(u[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1],float),np.nan)
  counts=np.isfinite(a).sum(axis=1)
  eq=np.divide(np.nansum(a,axis=1),counts,out=np.full(counts.shape,np.nan),where=counts>0)[kk]
  ch=np.stack([np.array(u[:,m['js'][0]:m['js'][-1]+1,f],float) for f in m['faces']])
  ch[~m['wet']]=np.nan
  tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,equator=eq,channel=ch);tmp.replace(dest)
  return eq,ch
 eqs=[];chs=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for n,(eq,ch) in enumerate(pool.map(read,src),1):
   eqs.append(eq);chs.append(ch)
   if n%12==0:print('NAS section months',n,'/336',flush=True)
 bp=np.stack(eqs);bu=np.stack(chs)
 np.savez_compressed(OUT/'extracted_sections.npz',equator=bp,channel=bu,time=time.values)
 common=np.isfinite(ctrl).all(axis=0)&np.isfinite(gl).all(axis=0)&np.isfinite(bp).all(axis=0)
 fields={'CTRL':ctrl,'BP':bp,'GL12':gl}
 filtered={k:filt(v,time.month.to_numpy())[36:-36] for k,v in fields.items()}
 std={k:v.std(axis=0) for k,v in filtered.items()}
 correlations={}
 for k in fields:
  a=filtered[k][:,common];b=filtered['GL12'][:,common]
  a=a-a.mean(axis=0);b=b-b.mean(axis=0)
  corr=np.full(common.shape,np.nan)
  corr[common]=(a*b).sum(axis=0)/np.sqrt((a*a).sum(axis=0)*(b*b).sum(axis=0))
  correlations[k]=corr
 z=-rd('RC',(40,))[kk];x=xall[ii];rows=[]
 for region,(lo,hi) in REGIONS.items():
  for band,(top,bot) in BANDS.items():
   dz=np.maximum(0,np.minimum(rf[kk+1],bot)-np.maximum(rf[kk],top));w=dz[:,None]*((x>=lo)&(x<=hi))[None]*common
   def avg(a):return float(np.sum(np.where(w>0,a,0)*w)/w.sum())
   for k in fields:
    rows.append(dict(region=region,band=band,dataset=k,rms_local_std_cm_s=np.sqrt(avg(std[k]**2))*100,amplitude_ratio_to_GL12=np.sqrt(avg(std[k]**2)/avg(std['GL12']**2)),weighted_mean_local_r_to_GL12=avg(correlations[k]),signed_mean_U_cm_s=avg(fields[k].mean(axis=0))*100))
 pd.DataFrame(rows).to_csv(OUT/'regional_amplitude.csv',index=False)
 # Same audited native channel extraction for reference and CTRL.
 with xr.open_dataset(ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport/source_sections.nc') as d:
  assert pd.DatetimeIndex(d.time.values).equals(time)
  cu=d.model_u.values;gu=d.reference_u.values
 with xr.open_dataset(ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc') as d:
  r=reference_geometry(d,Path('/public/home/zhanghang/data/GLO-MFC_001_030_coordinates.nc'),Path('/public/home/zhanghang/data/GLO-MFC_001_030_mask_bathy.nc'),2)
 qrows=[];series={'time':time};closure=[];localrows=[];area_errors=[]
 for name,top,bot in [('1900-bottom',1900,np.inf),('1900-3000',1900,3000),('1000-1900',1000,1900),('500-1000',500,1000)]:
  mm={**m,'bottom':np.minimum(m['bottom'],bot)};rr={**r,'bottom':np.minimum(r['bottom'],bot)}
  op=common_operator(mm,rr,top)
  ma=np.asarray(op.sum(axis=1)).reshape(m['top'].shape);ra=np.asarray(op.sum(axis=0)).reshape(r['top'].shape)
  area_errors.append(float(abs(ma.sum()-ra.sum())))
  area=ma.sum(axis=0);valid=area>0
  for key,u in [('CTRL',cu),('BP',bu)]:
   flux=np.where(ma[None]>0,u*ma[None],0).sum(axis=1)
   velocity=np.divide(flux,area[None],out=np.full_like(flux,np.nan),where=valid[None])
   f=filt(velocity,time.month.to_numpy())[36:-36]
   localrows.append(dict(band=name,dataset=key,common_area_km2=float(area.sum()/1e6),local_rms_std_cm_s=float(np.sqrt(np.sum(np.where(valid,f.var(axis=0),0)*area)/area.sum())*100)))
  native=np.maximum(0,mm['bottom']-np.maximum(mm['top'],top))*m['metric_per_degree'][:,None,:]*(m['north']-m['south'])[None,None,:]*m['face_weights'][:,None,None]
  qs={}
  for key,u,area in [('CTRL_common',cu,ma),('BP_common',bu,ma),('GL12_common',gu,ra),('CTRL_native',cu,native),('BP_native',bu,native)]:
   q,e,west=integrate(u,area);qs[key]=q;series[name+'_'+key]=q
   closure.append(float(np.max(abs(q-e-west))))
  fq={k:filt(v[:,None],time.month.to_numpy())[36:-36,0] for k,v in qs.items()}
  for key,q in qs.items():
   f=fq[key];ref=fq['GL12_common']
   same=key.endswith('_common')
   qrows.append(dict(band=name,dataset=key,mean_Sv=float(q.mean()),raw_monthly_std_Sv=float(q.std()),interannual_std_Sv=float(f.std()),correlation_GL12=float(np.corrcoef(f,ref)[0,1]) if same else np.nan,std_ratio_GL12=float(f.std()/ref.std()) if same else np.nan,anomaly_rmse_GL12_Sv=float(np.sqrt(np.mean(((f-f.mean())-(ref-ref.mean()))**2))) if same else np.nan))
   series[name+'_'+key+'_bp28']=np.r_[np.full(36,np.nan),f,np.full(36,np.nan)]
 pd.DataFrame(series).to_csv(OUT/'transport_series.csv',index=False);pd.DataFrame(qrows).to_csv(OUT/'transport_metrics.csv',index=False)
 pd.DataFrame(localrows).to_csv(OUT/'channel_local_amplitude.csv',index=False)
 fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
 for key,label in [('GL12_common','GL12'),('CTRL_common','CTRL'),('BP_common','Historical BP')]:
  q=series['1900-bottom_'+key];axs[0].plot(time,q,label=label,alpha=.8,lw=1)
  axs[1].plot(time[36:-36],series['1900-bottom_'+key+'_bp28'][36:-36],label=label)
 for ax in axs:ax.axhline(0,color='gray',lw=.6);ax.legend();ax.set_ylabel('Eastward transport (Sv)')
 axs[0].set_title('142°E coast to 2°N, below 1900 m | identical GL12-overlap wet area | monthly')
 axs[1].set_title('2–8 year anomalies | July 1999–June 2021')
 fig.savefig(FIG/'B01_transport.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(15,6),sharey=True,layout='constrained')
 table=pd.DataFrame(rows)
 for ax,region in zip(axs,list(REGIONS)[:3]):
  for k in fields:
   mask=common*((x>=REGIONS[region][0])&(x<=REGIONS[region][1]))[None]
   v=np.sqrt(np.sum(np.where(mask,std[k]**2,0),axis=1)/mask.sum(axis=1))*100
   ax.plot(v,z,label=k)
  ax.axhline(300,color='gray',ls='--');ax.axhline(1900,color='gray',ls=':');ax.set(title=region,xlabel='RMS local 2–8 year std (cm/s)',ylim=(3000,0));ax.legend()
 axs[0].set_ylabel('Depth (m)');fig.savefig(FIG/'B02_vertical_amplitude.png',dpi=180);plt.close(fig)
 xr.Dataset({**{k:(('depth','longitude'),np.where(common,v,np.nan)) for k,v in std.items()},**{k+'_correlation_GL12':(('depth','longitude'),v) for k,v in correlations.items()}},coords=dict(depth=z,longitude=x)).to_netcdf(OUT/'equatorial_std.nc')
 (OUT/'validation.json').write_text(json.dumps(dict(common_points=int(common.sum()),months=len(time),filter_support='1996-07 to 2024-06',statistics_period='1999-07 to 2021-06',directional_closure_max_Sv=max(closure),common_area_max_abs_error_m2=max(area_errors),config_warning='Historical BP also changes viscA4 and GGL90 ck; bulk wind affects heat as well as stress'),indent=2))
 print(pd.DataFrame(qrows).to_string(index=False),flush=True)
 print(table.query("band=='1900-3000'").to_string(index=False),flush=True)

if __name__=='__main__':main()
