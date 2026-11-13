"""Independent series checks and damping validation for completed 10-day factorial."""
from compare_ten_day_factorial import *

def main():
 times=pd.DatetimeIndex(np.load(DEST/'WA10_wave_components.npz')['time']);use=times>=pd.Timestamp('1996-01-01');mm=times[use].month.to_numpy();d=dict(np.load(DEST/'WA10_wave_components.npz'));fu=filt(d['u_points'][use],mm)[36:-36];fe=filt(d['ext_points'][use],mm)[36:-36];pred=-d['point_mask'][None]*fu/864000;inside=d['point_mask']>0;outside=~inside;relative=np.sqrt(np.mean((fe-pred)**2,axis=0))[inside]/np.sqrt(np.mean(pred**2,axis=0))[inside];assert relative.max()<1e-5;assert np.max(abs(fe[:,outside]))==0;assert (np.mean(fu*fe,axis=0)[inside]<0).all()
 ss=pd.read_csv(DEST/'filtered_series.csv',index_col=0);effects=pd.read_csv(DEST/'factorial_effects.csv');errors=[]
 for _,r in effects.query("kind=='low' and trim==36 and tau_days==10 and effect=='interaction'").iterrows():
  pre=f'low_{r.location}_{r.depth}_';v=(ss[pre+'WA10']-ss[pre+'W']-ss[pre+'A10']+ss[pre+'C']).values[36:-36];errors.append(abs(100*np.sqrt(np.mean(v*v))-r.rms_cm_s))
 assert max(errors)<1e-10
 # Independently replay all W and WA30 default layer metrics from the previous depth analysis.
 new=pd.read_csv(DEST/'velocity_metrics.csv');old=pd.read_csv(OUT/'depth_sensitivity/metrics.csv');replay=[]
 for _,r in new.query("trim==36 and experiment in ['W','WA30']").iterrows():
  key='WA' if r.experiment=='WA30' else 'W';x=old.query('kind==@r.kind and location==@r.location and depth==@r.depth and trim==36 and experiment==@key');replay.append(abs(r.std_cm_s-x.std_cm_s.iloc[0]))
 assert max(replay)<1e-6
 # One raw point/latitude slice checked against the independently stored extraction.
 g=grid(ROOT/'output'/RUNS['WA10']);p=ROOT/'output'/RUNS['WA10'];m=sorted(p.glob('diag3d.*.meta'))[200];nr,fl,t=meta(m);a=np.memmap(m.with_suffix('.data'),dtype='>f4',shape=(nr,NZ,NY,NX),mode='r');j=np.flatnonzero(abs(g['y'])<=.5);ix=np.argmin(abs(g['x']-142));z=20;wet=g['hw'][z,j,ix]>0;expected=float(np.mean(a[fl.index('UVEL'),z,j[wet],ix]));cached=float(np.load(DEST/'WA10_velocity.npz')['u'][200,z,ix]);assert abs(expected-cached)<1e-7
 val=json.loads((DEST/'validation.json').read_text());val.update(WA10_operator_max_relative_error=float(relative.max()),outside_operator_zero=True,anomaly_work_negative=True,interaction_series_replay_error_cm_s=max(errors),previous_W_WA30_replay_error_cm_s=max(replay),raw_sample_error_m_s=abs(expected-cached));dump(DEST/'validation.json',val)
 wave.DEST=DEST;basis=xr.load_dataset(ROOT/'research/model_control/evidence/vertical_mode_basis.nc');setup=wave.prepare(g,basis);quality=[]
 for key in ['W','WA10']:
  cache=dict(np.load(DEST/(key+'_wave_components.npz')));fcoef=filt(cache['coef'][use],mm)[36:-36];state=filt(cache['state_gates'][use],mm)[36:-36]
  for ci,(name,width,nr) in enumerate(wave.CONFIGS):
   for n in range(5):
    ig=int(np.argmin(abs(cache['gate_lons']-254)));ix=int(np.argmin(abs(cache['lon']-cache['gate_lons'][ig])));valid=setup['samplevalid'][(ci,ix)];B=setup['designs'][(ci,n)];obs=state[:,n,:,ig];pred=fcoef[:,ci,n,:nr+1,ix]@B.T;quality.append(dict(experiment=key,config=name,vertical_mode=n+1,longitude=cache['gate_lons'][ig],explained=1-np.sum((obs[:,valid]-pred[:,valid])**2)/np.sum(obs[:,valid]**2)))
 pd.DataFrame(quality).to_csv(DEST/'wave_fit_quality.csv',index=False)
 df=pd.read_csv(DEST/'wave_components.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for ax,comp in zip(axs,['K','R1']):
  for k,col in [('W','#0072B2'),('A10','#D55E00'),('WA10','#009E73')]:
   dd=df.query("config=='y7p5_R3' and component==@comp and experiment==@k");ax.plot(dd.vertical_mode,dd.ratio_C,'o-',label=k,color=col)
  ax.axhline(1,color='gray');ax.set(title=comp+' at 250–255 E',xlabel='Reference vertical mode',ylabel='Std / CTRL')
 axs[0].legend();fig.savefig(PLOT/'T03_wave_components.png',dpi=180);plt.close(fig);print('Independent validation passed',flush=True)
if __name__=='__main__':main()
