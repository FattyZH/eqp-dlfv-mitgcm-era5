"""Native-depth CTRL/GLORYS potential-temperature profiles and 2.2 C depths."""
from pathlib import Path
import sys, json, hashlib, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from analysis_common import ROOT, NX, NY, NZ, meta, sha, dump
DEST=ROOT/'research/temperature/2026-11-13_142E_equator'
PLOT=ROOT/'fig/261113_site_theta'
MODEL=ROOT/'output/261016_ctrl'
OBS=ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc'

def bracket(x,target):
 hi=int(np.searchsorted(x,target,side='right'));hi=min(max(hi,1),len(x)-1);lo=hi-1;w=(target-x[lo])/(x[hi]-x[lo]);assert 0<=w<=1
 return lo,hi,np.array([1-w,w])

def crossing(z,t):
 # Downward warm-to-cold crossing, restricting to deep ocean target.
 ok=np.isfinite(t[:-1])&np.isfinite(t[1:])&(z[:-1]>=500)&(t[:-1]>=2.2)&(t[1:]<=2.2)&(t[:-1]>t[1:]);ix=np.flatnonzero(ok)
 roots=[float(z[i]+(2.2-t[i])*(z[i+1]-z[i])/(t[i+1]-t[i])) for i in ix]
 # Adjacent brackets meeting an exact sample at threshold describe a single root.
 distinct=[]
 for r in roots:
  if not distinct or abs(r-distinct[-1])>1e-6:distinct.append(r)
 if len(distinct)!=1:return np.nan,len(distinct),None
 i=ix[0];return distinct[0],1,dict(z_above_m=float(z[i]),z_below_m=float(z[i+1]),theta_above=float(t[i]),theta_below=float(t[i+1]))

def main():
 DEST.mkdir(parents=True,exist_ok=True);PLOT.mkdir(parents=True,exist_ok=True)
 with xr.open_dataset(OBS) as ds:
  assert ds.thetao.attrs['standard_name']=='sea_water_potential_temperature'
  ix0,ix1,wx=bracket(ds.longitude.values,142);iy0,iy1,wy=bracket(ds.latitude.values,0)
  sub=ds.thetao.isel(longitude=slice(ix0,ix1+1),latitude=slice(iy0,iy1+1)).load()
  c=np.sum(sub.transpose('time','depth','latitude','longitude').values*wy[None,None,:,None]*wx[None,None,None,:],axis=(2,3));zc=ds.depth.values.astype(float);tc=pd.DatetimeIndex(ds.time.values).to_period('M').to_timestamp();obsattrs=dict(ds.thetao.attrs);coords=dict(longitude=ds.longitude.values[[ix0,ix1]].tolist(),latitude=ds.latitude.values[[iy0,iy1]].tolist(),longitude_weights=wx.tolist(),latitude_weights=wy.tolist())
 print('Loaded CMEMS temperature at native depths:',len(tc),'months',flush=True)
 x=np.fromfile(MODEL/'XC.data',dtype='>f4').reshape(NY,NX)[0].astype(float);y=np.fromfile(MODEL/'YC.data',dtype='>f4').reshape(NY,NX)[:,0].astype(float);zm=-np.fromfile(MODEL/'RC.data',dtype='>f4').astype(float);hc=np.fromfile(MODEL/'hFacC.data',dtype='>f4').reshape(NZ,NY,NX)
 ix0,ix1,wx=bracket(x,142);iy0,iy1,wy=bracket(y,0);wet=hc[:,iy0:iy1+1,ix0:ix1+1]>0
 dates=set(tc);sources=[]
 for m in sorted(MODEL.glob('diag3d.*.meta')):
  nr,fl,t=meta(m)
  if t in dates:sources.append((m,nr,fl,t))
 assert len(sources)==len(tc)
 signature=hashlib.sha256(json.dumps([(str(m),m.stat().st_mtime_ns,m.with_suffix('.data').stat().st_mtime_ns,m.with_suffix('.data').stat().st_size) for m,_,_,_ in sources]+[(str(OBS),OBS.stat().st_size,OBS.stat().st_mtime_ns)]).encode()).hexdigest()
 cache=DEST/'source_profiles.npz';tag=DEST/'source_signature.json'
 if cache.exists() and tag.exists() and json.loads(tag.read_text())['signature']==signature:
  m=np.load(cache)['model'];print('Reused source model profiles',flush=True)
 else:
  def read(item):
   path,nr,fields,t=item;a=np.memmap(path.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,NZ,NY,NX));v=np.array(a[fields.index('THETA'),:,iy0:iy1+1,:],float)[:,:,ix0:ix1+1];v[~wet]=np.nan;return np.sum(v*wy[None,:,None]*wx[None,None,:],axis=(1,2))
  rr=[]
  with ThreadPoolExecutor(max_workers=int(os.environ.get('EQP_IO_WORKERS','16'))) as pool:
   for i,v in enumerate(pool.map(read,sources)):
    rr.append(v)
    if (i+1)%60==0:print('CTRL profiles',i+1,flush=True)
  m=np.stack(rr);np.savez_compressed(cache,model=m,cmems=c,time=tc.values,model_depth=zm,cmems_depth=zc);dump(tag,dict(signature=signature))
 assert pd.DatetimeIndex([s[3] for s in sources]).equals(tc)
 rows=[];means=[];summary={}
 for label,start in [('common_full',tc[0]),('recent_2015_onward',pd.Timestamp('2015-01-01'))]:
  take=tc>=start;stats={}
  for key,z,a in [('ctrl',zm,m),('cmems',zc,c)]:
   aa=a[take];mu=np.mean(aa,axis=0);depth,count,br=crossing(z,mu);monthly=[crossing(z,t) for t in aa];v=np.array([t[0] for t in monthly]);assert np.isfinite(v).all(),(key,label,monthly)
   stats[key]=dict(isotherm_of_mean_profile_m=depth,mean_monthly_isotherm_m=float(v.mean()),std_monthly_isotherm_m=float(v.std()),min_monthly_m=float(v.min()),max_monthly_m=float(v.max()),percentiles_monthly_m={str(q):float(np.percentile(v,q)) for q in [5,10,50,90,95]},mean_profile_crossing_bracket=br,theta_at_1920_mean_C=float(np.interp(1920,z[np.isfinite(mu)],mu[np.isfinite(mu)])),multiple_or_missing_monthly_crossings=int(sum(n!=1 for _,n,_ in monthly)))
   for zz,theta in zip(z,mu):means.append(dict(period=label,source=key,depth_m=zz,mean_theta_C=theta))
  dz=stats['ctrl']['isotherm_of_mean_profile_m']-stats['cmems']['isotherm_of_mean_profile_m'];summary[label]=dict(first_month=str(tc[take][0].date()),last_month=str(tc[take][-1].date()),months=int(take.sum()),statistics=stats,ctrl_minus_cmems_mean_profile_isotherm_m=dz)
 for key,z,a in [('ctrl',zm,m),('cmems',zc,c)]:
  for t,aa in zip(tc,a):
   d,count,br=crossing(z,aa);rows.append(dict(time=t,source=key,isotherm_2p2_depth_m=d,crossing_count=count,theta_1920_C=float(np.interp(1920,z[np.isfinite(aa)],aa[np.isfinite(aa)]))))
 pd.DataFrame(rows).to_csv(DEST/'monthly_isotherms.csv',index=False);pd.DataFrame(means).to_csv(DEST/'mean_profiles.csv',index=False)
 # Numerical check: a threshold crossing is invariant under linear vertical interpolation
 # when the threshold's original bracketing interval is retained.
 check=[]
 for key,z,a in [('ctrl',zm,m),('cmems',zc,c)]:
  mu=np.mean(a,axis=0);d,n,br=crossing(z,mu);recovered=np.interp(d,z[np.isfinite(mu)],mu[np.isfinite(mu)]);check.append(abs(float(recovered)-2.2));assert check[-1]<1e-10
 info=dict(location=dict(longitude_E=142,latitude_N=0),model='261016_ctrl',reference=str(OBS.relative_to(ROOT)),variable_metadata=obsattrs,cmems_horizontal_interpolation=coords,model_horizontal_interpolation=dict(longitude=x[[ix0,ix1]].tolist(),latitude=y[[iy0,iy1]].tolist(),longitude_weights=wx.tolist(),latitude_weights=wy.tolist()),method='Bilinear horizontal interpolation of potential temperature at native vertical levels; linear interpolation within native bracketing levels to theta=2.2 C; no temporal filtering; common monthly means; native-level mean profile crossing is distinguished from mean monthly crossing.',results=summary,validation=dict(threshold_interpolation_max_error_C=max(check)),limitations=['Vertical interpolation is diagnostic, not meter-scale accuracy; native crossing brackets are reported.','Point comparison does not establish the isotherm depth along the whole island-to-2N section.','CMEMS local file is monthly Jul1996-Jun2024; no mooring temperature profile is included.'])
 dump(DEST/'summary.json',info)
 # Save native-depth time series without forcing one source onto the other vertical grid.
 xr.Dataset(dict(ctrl_theta=(('time','model_depth'),m),cmems_theta=(('time','cmems_depth'),c)),coords=dict(time=tc,model_depth=zm,cmems_depth=zc),attrs=dict(longitude=142.,latitude=0.,units='degrees_C',method=info['method'])).to_netcdf(DEST/'profiles.nc')
 df=pd.DataFrame(rows);fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
 for key,z,a,col in [('ctrl',zm,m,'#D55E00'),('cmems',zc,c,'black')]:
  mu=np.mean(a,axis=0);p10,p90=np.percentile(a,[10,90],axis=0);axs[0].plot(mu,z,'o-',ms=3,label=key,color=col);axs[0].fill_betweenx(z,p10,p90,color=col,alpha=.13);d=summary['common_full']['statistics'][key]['isotherm_of_mean_profile_m'];axs[0].plot([2.2],[d],'s',color=col);axs[0].annotate(f'{key}: {d:.0f} m',(2.2,d),xytext=(12,18 if key=='ctrl' else -25),textcoords='offset points',color=col,arrowprops=dict(arrowstyle='-',color=col,lw=.7))
  vv=df.query('source==@key');axs[1].plot(vv.time,vv.isotherm_2p2_depth_m,color=col,label=key,lw=1)
 axs[0].axvline(2.2,color='gray',ls='--');axs[0].axhline(1920,color='gray',ls=':',label='1920 m');axs[0].set(ylim=(3000,1000),xlim=(1,5),xlabel='Potential temperature (C)',ylabel='Depth (m)',title='Mean profile; shading: monthly 10–90%');axs[0].legend()
 axs[1].axhline(1920,color='gray',ls=':');axs[1].invert_yaxis();axs[1].set(ylabel='2.2 C isotherm depth (m)',title='Monthly isotherm depth; no filtering');axs[1].legend();fig.suptitle(f'142 E, 0 N | {tc[0]:%Y-%m} to {tc[-1]:%Y-%m}');fig.savefig(PLOT/'theta_profile_and_isotherm.png',dpi=180);plt.close(fig)
 dump(DEST/'manifest.json',dict(script_sha256=sha(Path(__file__)),source_signature=signature,artifacts=[str(q.relative_to(ROOT)) for q in sorted(DEST.iterdir()) if q.is_file()]))
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
