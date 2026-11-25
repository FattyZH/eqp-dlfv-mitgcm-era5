"""Additional paired-period KPP checks and phase-aligned CEOF comparisons."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'analysis/comparison'),str(ROOT/'analysis/mechanism')]
from compare_channel_transport import model_geometry,integrate
from compare_mooring_transport import sha
OUT=ROOT/'research/mechanism/2026-11-25_new_four';FIG=ROOT/'fig/261125_new_four';PRIOR=ROOT/'research/mechanism/2026-11-24_vector_invariant'
keys=['CTRL','GL12','VI_V3','VI_AV','Nr38','Nr38_gsm'];d={k:dict(np.load(OUT/(k+'_ceof.npz'))) for k in keys};mask=d['CTRL']['mask'];w=d['CTRL']['weights'];x=d['CTRL']['longitude'];z=d['CTRL']['depth']
rf=-np.fromfile(ROOT/'output/261016_ctrl1/RF.data','>f4').astype(float)
rc=-np.fromfile(ROOT/'output/261016_ctrl1/RC.data','>f4')
kdepth=np.flatnonzero(np.isin(rc,z))
rows=[]
for k,a in d.items():
 for i in range(4):
  overlap=np.sum(np.conj(d['CTRL']['spatial_mode'][i,mask])*a['spatial_mode'][i,mask]*w[mask]);rot=np.exp(-1j*np.angle(overlap)) if k!='CTRL' else 1.
  a['spatial_mode'][i]*=rot;a['time_coefficient'][:,i]*=np.conj(rot)
 for region,lo,hi,top in [('full',130,285,0),('west_deep',130,150,1900)]:
  dz=np.maximum(0,np.minimum(rf[kdepth+1],3000)-np.maximum(rf[kdepth],top))
  ww=dz[:,None]*mask*((x>=lo)&(x<=hi))[None];sel=ww>0
  for i in range(2):
   u=a['spatial_mode'][i,sel];v=d['GL12']['spatial_mode'][i,sel];wg=ww[sel]
   cong=abs(np.sum(np.conj(v)*u*wg))/np.sqrt(np.sum(abs(v)**2*wg)*np.sum(abs(u)**2*wg))
   tc=a['time_coefficient'][:,i];ref=d['GL12']['time_coefficient'][:,i]
   rows.append(dict(dataset=k,mode=i+1,region=region,variance_fraction=float(a['explained_variance'][i]),mode_rms_cm_s=float(np.sqrt(np.sum(abs(u)**2*wg)/wg.sum())*100),congruence_GL12=float(cong),TC_complex_correlation_GL12=float(abs(np.vdot(ref,tc))/np.sqrt(np.vdot(ref,ref).real*np.vdot(tc,tc).real))))
 np.savez_compressed(OUT/(k+'_ceof_aligned.npz'),**a)
pd.DataFrame(rows).to_csv(OUT/'ceof_comparison.csv',index=False)
fig,axs=plt.subplots(2,4,figsize=(19,8),layout='constrained',sharey=True)
for j,k in enumerate(['GL12','CTRL','Nr38','Nr38_gsm']):
 a=d[k]['spatial_mode'][0];im=axs[0,j].pcolormesh(x,z,abs(a)*100,shading='nearest',cmap='viridis',vmin=0,vmax=8);axs[0,j].set_title(f'{k}: CEOF1 {d[k]["explained_variance"][0]:.1%}')
 phase=axs[1,j].pcolormesh(x,z,np.where(mask,-np.angle(a)*180/np.pi,np.nan),shading='nearest',cmap='twilight',vmin=-180,vmax=180)
 for ax in axs[:,j]:ax.set(ylim=(3000,0),xlabel='Longitude E',ylabel='Depth m')
fig.colorbar(im,ax=list(axs[0]),label='Amplitude cm/s');fig.colorbar(phase,ax=list(axs[1]),label='Phase lag degrees, aligned to CTRL');fig.savefig(FIG/'CEOF1.png',dpi=160);plt.close(fig)
# Fixed common period for incomplete KPP, raw means only; no interannual ranking.
with np.load(OUT/'VI_AV_KPP_sources.npz') as c:early=pd.DatetimeIndex(c['time'])
with np.load(ROOT/'research/mechanism/2026-11-16_depth_amplitude/source_sections.npz') as c:ts=pd.DatetimeIndex(c['time']);kk=c['kk'];eq={'CTRL':c['model'][ts.get_indexer(early)]}
for k in ['VI_V3','VI_AV']:
 with np.load(PRIOR/(k+'_sources.npz')) as c:eq[k]=c['equator'][ts.get_indexer(early)]
for k in ['VI_KPP','VI_AV_KPP']:
 with np.load(OUT/(k+'_sources.npz')) as c:eq[k]=c['equator'][pd.DatetimeIndex(c['time']).get_indexer(early)][:,kk]
emask=np.logical_and.reduce([np.isfinite(a).all(0) for a in eq.values()]);raw=[]
qold=pd.read_csv(PRIOR/'native_transport.csv',index_col=0,parse_dates=True);qnew=pd.read_csv(OUT/'new_runs_available_native_Sv.csv',index_col=0,parse_dates=True)
for k,a in eq.items():
 v=(qnew if k in qnew else qold).loc[early,k]
 raw.append(dict(dataset=k,period_start=str(early[0].date()),period_end=str(early[-1].date()),months=len(early),mean_Q_Sv=float(v.mean()),raw_monthly_std_Sv=float(v.std(ddof=0))))
pd.DataFrame(raw).to_csv(OUT/'KPP_common_early_raw_metrics.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained',sharex=True,sharey=True)
for ax,k in zip(axs.flat,['VI_V3','VI_KPP','VI_AV','VI_AV_KPP']):
 a=np.where(emask,eq[k].mean(0)*100,np.nan);im=ax.pcolormesh(x,z,a,shading='nearest',cmap='RdBu_r',vmin=-60,vmax=60);ax.set(title=k,ylim=(3000,0),xlabel='Longitude E',ylabel='Depth m')
fig.colorbar(im,ax=list(axs.flat),label='Mean U cm/s');fig.suptitle(f'Common incomplete-run window: {early[0]:%Y-%m} to {early[-1]:%Y-%m}, {len(early)} months; raw mean only');fig.savefig(FIG/'KPP_early_mean_U.png',dpi=160);plt.close(fig)
# Audit shared file dependencies and exact Nr38 versus Nr38_gsm configuration differences.
audit=json.loads((OUT/'configuration_audit.json').read_text())
import f90nml
A=ROOT/'output/261124_Nr38';B=ROOT/'output/261124_Nr38_gsm';ca=f90nml.read(A/'data');cb=f90nml.read(B/'data')
diff={g:{k:[ca.get(g,{}).get(k),cb[g].get(k)] for k in set(ca.get(g,{}))|set(cb[g]) if ca.get(g,{}).get(k)!=cb[g].get(k)} for g in cb};diff={g:v for g,v in diff.items() if v}
pair=dict(data_differences=diff,identical_files={k:sha(A/k)==sha(B/k) for k in ['data.exf','data.obcs','data.pkg','data.ggl90','RF.data','RC.data']},topography_identical=sha(A/'bathy_avg.bin')==sha(B/'bathy_gsm.bin'),note='Initial/OBCS paths are shared mutable symlinks; current hashes cannot establish launch-time content.')
for key in ['VI_KPP','VI_AV_KPP']:
 p=ROOT/'output'/audit[key]['experiment']/'obcs';audit[key]['current_boundary_dependency']=str(p.resolve());audit[key]['required_OBNu_4_40_exists_now']=(p/'OBNu_4_40.bin').exists()
(OUT/'configuration_audit.json').write_text(json.dumps(audit,indent=2));(OUT/'Nr38_pair_audit.json').write_text(json.dumps(pair,indent=2))
# Reproduce original CTRL integration with generalized vertical-size geometry.
m=model_geometry(ROOT/'output/261016_ctrl1',2);area=np.maximum(0,m['bottom']-np.maximum(m['top'],1900))*m['metric_per_degree'][:,None,:]*(m['north']-m['south'])[None,None,:]*m['face_weights'][:,None,None]
with xr.open_dataset(ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport/source_sections.nc') as c:v=integrate(c.model_u.values,area)[0]
error=float(np.max(abs(v-qold.CTRL.values)));assert error<1e-10
validation=json.loads((OUT/'validation.json').read_text());validation['CTRL_native_integration_reproduction_max_error_Sv']=error
validation['old_observation_metrics_reproduction_max_error']=float(np.max(abs(pd.read_csv(OUT/'recent_obs_metrics.csv').set_index('dataset').loc[['CTRL','VI_V3','VI_AV'],'correlation'].values-np.array([.3055578679402394,.25406426628941237,.23980848213937977]))))
(OUT/'validation.json').write_text(json.dumps(validation,indent=2))
print(pd.DataFrame(rows).query('region=="west_deep" and mode==1').to_string(index=False));print(pd.DataFrame(raw).to_string(index=False));print('pair',pair);print('validation',validation)
with xr.open_dataset(OUT/'equatorial_statistics.nc') as ds:
 fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained',sharex=True,sharey=True)
 for ax,k in zip(axs.flat,['GL12','CTRL','Nr38','Nr38_gsm']):
  im=ax.pcolormesh(x,z,ds[k+'_std'],shading='nearest',cmap='viridis',vmin=0,vmax=2.5)
  ax.set(title=k,ylim=(3000,1000),xlabel='Longitude E',ylabel='Depth m');ax.axhline(1900,color='white',ls='--',lw=.7)
 fig.colorbar(im,ax=list(axs.flat),label='2-8yr local standard deviation cm/s')
 fig.savefig(FIG/'interannual_std_deep.png',dpi=160);plt.close(fig)
