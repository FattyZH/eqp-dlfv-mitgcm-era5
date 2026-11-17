"""Matched-area142E channel amplitude and cancellation diagnostics,2-8years."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/comparison'))
from compare_channel_transport import model_geometry,reference_geometry,common_operator,bandpass
DEST=ROOT/'research/mechanism/2026-11-16_depth_amplitude'
PLOT=ROOT/'fig/261116_depth_amplitude'
SOURCE=ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport'

def filter_matrix(a,months):
 out=np.full_like(a,np.nan,dtype=float);wet=np.isfinite(a).all(axis=0)
 out[:,wet]=np.stack([bandpass(a[:,wet][:,i],months) for i in range(wet.sum())],axis=1)
 return out

def main():
 m=model_geometry(ROOT/'output/261016_ctrl1',2)
 with xr.open_dataset(ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc') as ds:
  r=reference_geometry(ds,Path('/public/home/zhanghang/data/GLO-MFC_001_030_coordinates.nc'),Path('/public/home/zhanghang/data/GLO-MFC_001_030_mask_bathy.nc'),2)
 with xr.open_dataset(SOURCE/'source_sections.nc') as ds:
  mu=ds.model_u.values;ru=ds.reference_u.values;time=pd.DatetimeIndex(ds.time.values)
 rows=[];validation=[];section=None;series={}
 for name,lo,hi in [('0-300',0,300),('500-1000',500,1000),('1000-1900',1000,1900),('1900-3000',1900,3000),('1900-availablebottom',1900,np.inf)]:
  mm={**m,'bottom':np.minimum(m['bottom'],hi)};rr={**r,'bottom':np.minimum(r['bottom'],hi)}
  op=common_operator(mm,rr,lo);ma=np.array(op.sum(axis=1)).reshape(m['top'].shape);ra=np.array(op.sum(axis=0)).reshape(r['top'].shape)
  a=ma.sum(axis=0);valid=a>0
  mflux=np.where(ma[None]>0,mu*ma[None],0).sum(axis=1)
  rflux=np.asarray(op@np.where(np.isfinite(ru),ru,0).reshape(len(time),-1).T).T.reshape(mu.shape).sum(axis=1)
  mv=np.divide(mflux,a[None],out=np.full_like(mflux,np.nan),where=valid[None]);rv=np.divide(rflux,a[None],out=np.full_like(rflux,np.nan),where=valid[None])
  fm=filter_matrix(mv,time.month.to_numpy())[36:-36];fr=filter_matrix(rv,time.month.to_numpy())[36:-36]
  vm=np.var(fm,axis=0);vr=np.var(fr,axis=0)
  wm=np.sum(np.where(valid,vm,0)*a)/a.sum();wr=np.sum(np.where(valid,vr,0)*a)/a.sum()
  qm=np.where(valid[None],fm*a[None],0).sum(axis=1)/1e6;qr=np.where(valid[None],fr*a[None],0).sum(axis=1)/1e6
  # Longitude-flux projection preserves integrated transport exactly.
  rawm=mflux.sum(axis=(1,2))/1e6;rawr=rflux.sum(axis=(1,2))/1e6
  errm=float(abs(qm.sum(axis=1)-bandpass(rawm,time.month.to_numpy())[36:-36]).max());errr=float(abs(qr.sum(axis=1)-bandpass(rawr,time.month.to_numpy())[36:-36]).max())
  if max(errm,errr)>1e-9:raise AssertionError('Filtering/integration closure')
  for key,q,local in [('CTRL',qm,wm),('GL12',qr,wr)]:
   cov=np.cov(q,rowvar=False,ddof=0);total=float(np.var(q.sum(axis=1)));diag=float(np.trace(cov));off=total-diag
   rows.append(dict(band=name,dataset=key,common_area_km2=float(a.sum()/1e6),local_rms_std_cm_s=float(np.sqrt(local)*100),net_transport_std_Sv=float(np.sqrt(total)),sum_lat_bin_variances_Sv2=diag,latitude_covariance_sum_Sv2=off,total_transport_variance_Sv2=total))
   series[(name,key)]=q.sum(axis=1)
  validation.append(dict(band=name,model_filter_flux_closure_error_Sv=errm,reference_filter_flux_closure_error_Sv=errr))
  if name=='1900-availablebottom':section=dict(area=a,model_mean=mv.mean(axis=0),reference_mean=rv.mean(axis=0),model_std=np.std(fm,axis=0),reference_std=np.std(fr,axis=0))
 table=pd.DataFrame(rows);table.to_csv(DEST/'channel_matched_area_metrics.csv',index=False)
 pd.DataFrame(series,index=time[36:-36]).to_csv(DEST/'channel_bandpass_series.csv',index_label='time')
 xr.Dataset({k:(('depth','latitude'),v) for k,v in section.items()},coords=dict(depth=m['depth'],latitude=m['lat']),attrs=dict(definition='Two bracketing Uface overlap fluxes weighted then divided by common area',filter='2-8year1999-07 to2021-06;support1996-07 to2024-06',source='Previously audited native UVEL andGL12 uo source_sections.nc')).to_netcdf(DEST/'channel_matched_area_section.nc')
 (DEST/'channel_matched_area_validation.json').write_text(json.dumps(validation,indent=2))
 fig,axs=plt.subplots(1,3,figsize=(13,6),sharey=True,layout='constrained')
 for ax,k,title in zip(axs,['model_std','reference_std','ratio'],['CTRL2-8year std','GL12 matched-area std','CTRL/GL12 amplitude ratio']):
  v=section[k]*100 if k!='ratio' else section['model_std']/section['reference_std']
  im=ax.pcolormesh(m['lat'],m['depth'],v,shading='nearest',vmin=0,vmax=3 if k!='ratio' else 1.5,cmap='viridis' if k!='ratio' else 'RdBu_r');ax.set(title=title,xlabel='Latitude N',ylim=(3500,1800));fig.colorbar(im,ax=ax,label='cm/s' if k!='ratio' else 'ratio')
 axs[0].set_ylabel('Depth (m)');fig.suptitle('142E below1900m | identical overlap wet area |1999-2021')
 fig.savefig(PLOT/'D04_channel_latitude_depth.png',dpi=180);plt.close(fig)
 print(table.to_string(index=False),flush=True)

if __name__=='__main__':main()
