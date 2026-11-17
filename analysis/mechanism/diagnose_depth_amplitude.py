"""Locate CTRL/GL12 interannual amplitude deficits in equatorial sections.

Validated retained source checkpoint: original ±0.5degree latitude mean,
1996-07 to2024-06, common model-depth/longitude grid. Mean flow is kept
separate from2-8year standard deviation. Descriptive source/path diagnosis,
not a direct energy-budget or vertical-propagation attribution.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/comparison'))
from compare_channel_transport import meta
DEST=ROOT/'research/mechanism/2026-11-16_depth_amplitude'
PLOT=ROOT/'fig/261116_depth_amplitude'
SOURCE=Path('/tmp/eqp_final_ctrl_audit_sources.npz')
OLDKEY='261016_ck_0p07_a4_4e10'
FOLDER=ROOT/'output/261016_ctrl1'
REGIONS={'west130-150':(130,150),'central170-220':(170,220),'east220-255':(220,255),'eastern250-255':(250,255)}
BANDS={'0-100':(0,100),'100-300':(100,300),'300-500':(300,500),'500-1000':(500,1000),'1000-1900':(1000,1900),'1900-3000':(1900,3000)}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def filt(a,months):
 b=a.copy();mask=np.isfinite(b).all(axis=0);result=np.full(b.shape,np.nan)
 values=b[:,mask]
 for mon in range(1,13):values[months==mon]-=values[months==mon].mean(axis=0)
 result[:,mask]=sosfiltfilt(butter(4,[1/8,1/2],fs=12,btype='bandpass',output='sos'),values,axis=0)
 return result

def wmean(v,w):return np.sum(np.where(w>0,v,0)*w)/w.sum()
def main():
 DEST.mkdir(parents=True,exist_ok=True);PLOT.mkdir(parents=True,exist_ok=True)
 p=DEST/'source_sections.npz'
 if p.exists():
  with np.load(p) as d:m=d['model'];c=d['gl12'];time=pd.DatetimeIndex(d['time']);kk=d['kk']
 else:
  with np.load(SOURCE) as d:m=d[OLDKEY];c=d['cmems'];time=pd.DatetimeIndex(d['months']);kk=d['kk']
  np.savez_compressed(p,model=m,gl12=c,time=time.values,kk=kk)
 def rd(name,shape):return np.fromfile(FOLDER/(name+'.data'),dtype='>f4').reshape(shape)
 xall=rd('XG',(192,720))[0];yall=rd('YC',(192,720))[:,0];z=-rd('RC',(40,))[kk];rf=-rd('RF',(41,)).astype(float)
 ii=np.flatnonzero((xall>=130)&(xall<=285));jj=np.flatnonzero(abs(yall)<=.5);x=xall[ii]
 if m.shape!=(336,len(kk),len(ii)) or not time.equals(pd.date_range('1996-07-01',periods=336,freq='MS')):raise ValueError('Unexpected source checkpoint shape/date')
 # Revalidate retained cache against current actual source at first/middle/last
 # common dates; never infer identity solely from renamed experiment labels.
 hw=rd('hFacW',(40,192,720))[:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1]>0
 paths={meta(q)[2]:q for q in FOLDER.glob('diag3d.*.meta')};model_errors=[];reference_errors=[]
 for index in [0,168,335]:
  path=paths[time[index]];nr,fields,_=meta(path);raw=np.memmap(path.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720))
  u=np.where(hw,raw[fields.index('UVEL'),:,jj[0]:jj[-1]+1,ii[0]:ii[-1]+1],np.nan)
  counts=np.isfinite(u).sum(axis=1);v=np.divide(np.nansum(u,axis=1),counts,out=np.full(counts.shape,np.nan),where=counts>0)[kk]
  valid=np.isfinite(v)&np.isfinite(m[index]);model_errors.append(float(np.max(abs(v[valid]-m[index][valid]))))
  with xr.open_dataset(ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc') as ds:
   v=ds.uo.isel(time=index).sel(latitude=slice(-.5,.5)).mean('latitude').load().interp(depth=z,longitude=x).transpose('depth','longitude').values
  valid=np.isfinite(v)&np.isfinite(c[index]);reference_errors.append(float(np.max(abs(v[valid]-c[index][valid]))))
 if max(model_errors)>1e-6 or max(reference_errors)>1e-6:raise AssertionError('Retained source no longer matches actual inputs')
 common=np.isfinite(m).all(axis=0)&np.isfinite(c).all(axis=0)
 mf=filt(m,time.month.to_numpy());cf=filt(c,time.month.to_numpy())
 mainperiod=slice(36,-36);fm=mf[mainperiod];fc=cf[mainperiod];dates=time[mainperiod]
 mu=np.where(common,m.mean(axis=0),np.nan);cu=np.where(common,c.mean(axis=0),np.nan)
 sm=np.std(fm,axis=0);sc=np.std(fc,axis=0);ratio=np.divide(sm,sc,out=np.full(sm.shape,np.nan),where=common&(sc>1e-8))
 corr=np.full(sm.shape,np.nan);corr[common]=np.sum((fm[:,common]-fm[:,common].mean(axis=0))*(fc[:,common]-fc[:,common].mean(axis=0)),axis=0)/np.sqrt(np.sum((fm[:,common]-fm[:,common].mean(axis=0))**2,axis=0)*np.sum((fc[:,common]-fc[:,common].mean(axis=0))**2,axis=0))
 xr.Dataset(dict(model_mean=(('depth','longitude'),mu),gl12_mean=(('depth','longitude'),cu),model_interannual_std=(('depth','longitude'),sm),gl12_interannual_std=(('depth','longitude'),sc),std_ratio=(('depth','longitude'),ratio),correlation=(('depth','longitude'),corr),common_mask=(('depth','longitude'),common)),coords=dict(depth=z,longitude=x),attrs=dict(velocity_units='m/s',mean_period='1996-07 to2024-06',variability_period='1999-07 to2021-06',latitude='±0.5degree wet-point mean; source-specific sampling before common section mask',filter='Own monthly climate removed; fourthorder zerophase2-8years;36months trim')).to_netcdf(DEST/'section_statistics.nc')
 rows=[];profiles=[]
 for region,(lo,hi) in REGIONS.items():
  xx=(x>=lo)&(x<=hi)
  for k,depth in enumerate(z):
   mask=common[k]&xx
   if not mask.any():continue
   a=float(np.sqrt(np.mean(sm[k,mask]**2)));b=float(np.sqrt(np.mean(sc[k,mask]**2)))
   profiles.append(dict(region=region,depth_m=float(depth),model_rms_std_cm_s=a*100,gl12_rms_std_cm_s=b*100,std_ratio=a/b,model_signed_mean_cm_s=float(np.mean(mu[k,mask])*100),gl12_signed_mean_cm_s=float(np.mean(cu[k,mask])*100),longitudes=int(mask.sum())))
  for band,(top,bottom) in BANDS.items():
   dz=np.maximum(0,np.minimum(rf[kk+1],bottom)-np.maximum(rf[kk],top));w=common*xx[None]*dz[:,None]
   if w.sum()==0:continue
   a=float(np.sqrt(wmean(sm**2,w)));b=float(np.sqrt(wmean(sc**2,w)))
   ms=np.sum(np.where(w[None]>0,fm,0)*w[None],axis=(1,2))/w.sum();cs=np.sum(np.where(w[None]>0,fc,0)*w[None],axis=(1,2))/w.sum()
   rows.append(dict(region=region,band=band,model_rms_std_cm_s=a*100,gl12_rms_std_cm_s=b*100,local_amplitude_ratio=a/b,coherent_mean_std_model_cm_s=float(ms.std()*100),coherent_mean_std_gl12_cm_s=float(cs.std()*100),coherent_mean_amplitude_ratio=float(ms.std()/cs.std()),model_signed_mean_cm_s=float(wmean(mu,w)*100),gl12_signed_mean_cm_s=float(wmean(cu,w)*100),model_mean_flow_spatial_rms_cm_s=float(np.sqrt(wmean(mu**2,w))*100),gl12_mean_flow_spatial_rms_cm_s=float(np.sqrt(wmean(cu**2,w))*100)))
 table=pd.DataFrame(rows);table.to_csv(DEST/'regional_depth_metrics.csv',index=False);pd.DataFrame(profiles).to_csv(DEST/'regional_vertical_profiles.csv',index=False)
 # Same vertical reference basis for both datasets: compare modal amplitudes
 # before mixing vertical structure and phase cancellation into point signals.
 with xr.open_dataset(ROOT/'research/model_control/evidence/vertical_mode_basis.nc') as ds:
  basis=ds.load()
 if not np.allclose(basis.depth.values,z,atol=1e-3):raise ValueError('Mode basis grid differs')
 phi=basis.velocity_mode.values[:,0:6];quad=basis.quadrature_weight.values
 projector=(phi*quad[:,None]).T/quad.sum();complete=common.all(axis=0)
 mm=projector@fm[:,:,complete];cc=projector@fc[:,:,complete] # numpy broadcast matrix products:time,mode,longitude
 mcstd=mm.std(axis=0);ccstd=cc.std(axis=0);xc=x[complete]
 modal=[]
 for region,(lo,hi) in REGIONS.items():
  mask=(xc>=lo)&(xc<=hi)
  for n in range(6):
   a=float(np.sqrt(np.mean(mcstd[n,mask]**2)));b=float(np.sqrt(np.mean(ccstd[n,mask]**2)))
   modal.append(dict(region=region,mode=n,model_amplitude=a,gl12_amplitude=b,ratio=a/b,complete_longitudes=int(mask.sum())))
 pd.DataFrame(modal).to_csv(DEST/'modal_amplitude_ratios.csv',index=False)
 # Variance identity at each(z,x), then depth/longitude weighting. Distinguish
 # weak modal amplitudes from destructive interference and projection residual.
 partitions=[];closure=[]
 for dataset,coeff,field in [('CTRL',mm,fm[:,:,complete]),('GL12',cc,fc[:,:,complete])]:
  recon=phi@coeff
  rawvar=field.var(axis=0);lowvar=recon.var(axis=0)
  diagonal=phi**2@coeff.var(axis=0)
  residualvar=(field-recon).var(axis=0)
  crossmodal=lowvar-diagonal;crossresidual=rawvar-lowvar-residualvar
  closure.append(float(np.nanmax(abs(rawvar-(diagonal+crossmodal+residualvar+crossresidual)))))
  for region,(lo,hi) in REGIONS.items():
   mask=(xc>=lo)&(xc<=hi)
   for band,(top,bottom) in BANDS.items():
    dz=np.maximum(0,np.minimum(rf[kk+1],bottom)-np.maximum(rf[kk],top))
    w=dz[:,None]*mask[None]
    partitions.append(dict(dataset=dataset,region=region,band=band,
                           raw_variance_cm2_s2=float(wmean(rawvar,w)*1e4),
                           low_mode_variance_cm2_s2=float(wmean(lowvar,w)*1e4),
                           modal_diagonal_sum_cm2_s2=float(wmean(diagonal,w)*1e4),
                           intermodal_covariance_sum_cm2_s2=float(wmean(crossmodal,w)*1e4),
                           residual_variance_cm2_s2=float(wmean(residualvar,w)*1e4),
                           low_residual_covariance_sum_cm2_s2=float(wmean(crossresidual,w)*1e4)))
 pd.DataFrame(partitions).to_csv(DEST/'variance_partition.csv',index=False)
 (DEST/'variance_partition_validation.json').write_text(json.dumps(dict(max_identity_error_m2_s2=max(closure),common_complete_longitudes=int(complete.sum()),definition='Pointwise temporal variance, spatial weighted mean; signed covariance sums, not energy conversion'),indent=2))
 xr.Dataset(dict(model_std=(('mode','longitude'),mcstd),gl12_std=(('mode','longitude'),ccstd)),coords=dict(mode=np.arange(6),longitude=xc)).to_netcdf(DEST/'modal_longitude_amplitude.nc')
 # Fixed142E longitude interpolation at each depth; excludes unavailable coast
 # or one-sided wet points rather than filling zeros.
 i=int(np.searchsorted(x,142));alpha=(142-x[i-1])/(x[i]-x[i-1]);site_mask=common[:,i-1]&common[:,i]
 site_m=(1-alpha)*fm[:,:,i-1]+alpha*fm[:,:,i];site_c=(1-alpha)*fc[:,:,i-1]+alpha*fc[:,:,i]
 pd.DataFrame(dict(depth_m=z,common=site_mask,model_std_cm_s=np.where(site_mask,site_m.std(axis=0)*100,np.nan),gl12_std_cm_s=np.where(site_mask,site_c.std(axis=0)*100,np.nan),model_mean_cm_s=((1-alpha)*mu[:,i-1]+alpha*mu[:,i])*100,gl12_mean_cm_s=((1-alpha)*cu[:,i-1]+alpha*cu[:,i])*100)).to_csv(DEST/'section142E_equatorial_band_profile.csv',index=False)
 # Diagnostic comparison of normalized depth-to-upper amplitudes; these are
 # patterns, not estimates of vertical wave attenuation or a source location.
 ratios=[]
 for region in REGIONS:
  g=table[table.region==region].set_index('band');upper=g.loc['100-300','local_amplitude_ratio'];deep=g.loc['1900-3000','local_amplitude_ratio']
  ratios.append(dict(region=region,upper100_300_ratio=float(upper),deep1900_3000_ratio=float(deep),relative_deep_to_upper_ratio=float(deep/upper)))
 pd.DataFrame(ratios).to_csv(DEST/'relative_depth_deficit.csv',index=False)
 fig,axs=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True,layout='constrained')
 fields=[(mu*100,'CTRL mean U','RdBu_r',-20,20),(cu*100,'GL12 mean U','RdBu_r',-20,20),((mu-cu)*100,'Mean bias CTRL - GL12','RdBu_r',-10,10),(sm*100,'CTRL2-8year std','viridis',0,10),(sc*100,'GL12 2-8year std','viridis',0,10),(ratio,'Std ratio CTRL / GL12','RdBu_r',0,2)]
 for ax,(v,title,cmap,vmin,vmax) in zip(axs.flat,fields):
  im=ax.pcolormesh(x,z,v,cmap=cmap,vmin=vmin,vmax=vmax,shading='nearest');ax.set(title=title,xlabel='Longitude E',ylabel='Depth (m)',ylim=(3000,0));ax.axvline(142,color='gray',ls='--',lw=.7);ax.axhline(1900,color='gray',ls=':',lw=.7);fig.colorbar(im,ax=ax,label='ratio' if title.startswith('Std ratio') else 'cm/s')
 fig.suptitle('Equatorial band±0.5degree | common source1996-2024 | interannual1999-2021')
 fig.savefig(PLOT/'D01_mean_and_interannual_sections.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,4,figsize=(15,7),sharey=True,layout='constrained')
 for ax,region in zip(axs,REGIONS):
  p=pd.DataFrame(profiles).query('region==@region');ax.plot(p.model_rms_std_cm_s,p.depth_m,label='CTRL');ax.plot(p.gl12_rms_std_cm_s,p.depth_m,label='GL12');ax.set(title=region,xlabel='RMS local2-8year std (cm/s)',ylim=(3000,0));ax.axhline(1900,color='gray',ls=':',lw=.7);ax.legend()
 axs[0].set_ylabel('Depth (m)');fig.savefig(PLOT/'D02_vertical_amplitude_profiles.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for n in range(1,6):
  # Longitude smoothing for display only; CSV/metrics use native points.
  rr=pd.Series(mcstd[n]/ccstd[n],index=xc).rolling(21,center=True,min_periods=1).mean();axs[0].plot(xc,rr,label=f'n={n}')
 for region in REGIONS:
  p=table.query('region==@region');axs[1].plot(np.arange(len(BANDS)),p.local_amplitude_ratio,'o-',label=region)
 axs[0].axhline(1,color='gray',ls='--');axs[0].set(xlabel='Longitude E',ylabel='Modal std ratio CTRL/GL12',title='Common reference modes;5.25degree display mean');axs[0].legend(ncol=3)
 axs[1].axhline(1,color='gray',ls='--');axs[1].set_xticks(np.arange(len(BANDS)),BANDS.keys(),rotation=35);axs[1].set(ylabel='Local amplitude ratio CTRL/GL12',title='Depth-band comparison');axs[1].legend(fontsize=8)
 fig.savefig(PLOT/'D03_depth_and_modal_ratios.png',dpi=180);plt.close(fig)
 validation=dict(current_source_model_max_errors_m_s=model_errors,current_source_gl12_max_errors_m_s=reference_errors,common_points=int(common.sum()),complete_columns=int(complete.sum()),main_filter_period=[str(dates[0].date()),str(dates[-1].date())],all_means_use_unfiltered_common_source_period=True,mean_and_variability_not_interchanged=True,model_grid_sha256={n:sha(FOLDER/n) for n in ['XG.data','YC.data','RC.data','RF.data','hFacW.data']},source_checkpoint_sha256=sha(DEST/'source_sections.npz'))
 (DEST/'validation.json').write_text(json.dumps(validation,indent=2));(DEST/'summary.json').write_text(json.dumps(dict(source='Validated retained source arrays, mirrored to persistent research directory',latitude_band='±0.5degree',source_period='1996-07 to2024-06',filter_period='1999-07 to2021-06',filter='2-8year fourthorder zerophase;36month trim',relative_depth_deficit=ratios,limits=['Not an estimate of true source excitation or energy flux','Mean flow is background, not interannual amplitude','Equatorial latitude means are not full channel transport','Common reference modes are not local eigenmodes','Depth-to-source geographic normalization alone cannot distinguish distributed forcing and interference']),indent=2));(DEST/'manifest.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),source_arrays_sha256=sha(DEST/'source_sections.npz'),gl12_file_sha256=sha(ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc')),indent=2))
 print(table[['region','band','local_amplitude_ratio','coherent_mean_amplitude_ratio']].to_string(index=False),flush=True)
 print(pd.DataFrame(modal).to_string(index=False),flush=True)
 print('VALIDATION',json.dumps(validation),flush=True)

if __name__=='__main__':main()
