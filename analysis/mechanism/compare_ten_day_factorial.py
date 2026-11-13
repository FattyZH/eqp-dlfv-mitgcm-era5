"""Complete 10-day wind-bandstop x eastern absorber factorial; read-only sources."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from analysis_common import *
import equatorial_wave as wave
DEST=ROOT/'research/mechanism/2026-11-13_ten_day_factorial'
PLOT=ROOT/'fig/261113_ten_day_factorial'
STRENGTH=RETAINED/'research/mechanism/2026-10-21_absorber_strength'
RUNS={'C':'261016_ctrl','W':'261019_noiav_wind','A10':'261020_eabs_10d','WA10':'261021_noiav_wind_eabs_10d'}
DEPTHS={'500-1000':(500,1000),'1000-2000':(1000,2000),'1000-3000':(1000,3000),'2000-3000':(2000,3000)}

def audit_new():
 p=ROOT/'output'/RUNS['WA10'];ref=ROOT/'output'/'261019_noiav_wind_eabs';old=json.loads((OUT/'audit.json').read_text())['experiments']['WA'];hashes={f:sha(p/f)==h for f,h in old['hashes'].items()};assert all(hashes.values())
 diffs=[]
 def walk(a,b,path):
  if isinstance(a,dict) and isinstance(b,dict):
   for k in a.keys()|b.keys():walk(a.get(k),b.get(k),path+'/'+k)
  elif a!=b:diffs.append(dict(path=path,WA30=a,WA10=b))
 for f in ['data','data.cal','data.exf','data.pkg','data.ggl90','data.obcs','data.rbcs','data.diagnostics']:walk(f90nml.read(ref/f).todict(),f90nml.read(p/f).todict(),f)
 assert {d['path'] for d in diffs}=={'data.rbcs/rbcs_parm01/taurelax'+v for v in 'tsuv'},diffs
 assert all(d['WA30']==2629800 and d['WA10']==864000 for d in diffs)
 targets={f:sha(p/f)==v['sha256'] for f,v in old['rbcs_files'].items()};assert all(targets.values())
 for prefix in ['diag3d','diag2d','diag_mom_u']:
  times=[]
  for m in sorted(p.glob(prefix+'.*.meta')):
   nr,fl,t=meta(m);times.append(t);assert m.with_suffix('.data').stat().st_size==nr*NX*NY*(1 if prefix=='diag2d' else NZ)*4
  assert pd.DatetimeIndex(times).equals(pd.date_range('1991-01-01',periods=420,freq='MS'))
 assert 'Execution ended Normally' in tail(p/'STDOUT.0000')
 dump(DEST/'audit.json',dict(common_hash_matches=hashes,target_hash_matches=targets,namelist_differences=diffs,normal_end=True,months=420,limitation='Present shared target hashes do not independently establish historical immutability.'))
 print('WA10 audit passed: only four tau changes from WA30',flush=True)

def extract_new(g):
 p=ROOT/'output'/RUNS['WA10'];dest=DEST/'WA10_velocity.npz';tag=DEST/'WA10_velocity_sources.json';sig=signature(p)
 if dest.exists() and tag.exists() and json.loads(tag.read_text())['signature']==sig:return dict(np.load(dest))
 js=np.flatnonzero(abs(g['y'])<=.5);sl=slice(js[0],js[-1]+1);wet=g['hw'][:,sl]>0;j=int(np.argmin(abs(g['y'])));jl=int(np.where(js==j)[0][0]);i=int(np.searchsorted(g['x'],142)-1);a=(142-g['x'][i])/(g['x'][i+1]-g['x'][i])
 def read(m):
  nr,fl,t=meta(m);v=np.memmap(m.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,NZ,NY,NX));u=np.array(v[fl.index('UVEL'),:,sl,:],float);u[~wet]=np.nan;count=wet.sum(axis=1);eq=np.divide(np.nansum(u,axis=1),count,out=np.full((NZ,NX),np.nan),where=count>0)
  return t.to_datetime64(),eq.astype(np.float32),u[:,jl,i]*(1-a)+u[:,jl,i+1]*a
 rr=[]
 with ThreadPoolExecutor(max_workers=int(os.environ.get('EQP_IO_WORKERS','16'))) as pool:
  for n,r in enumerate(pool.map(read,sorted(p.glob('diag3d.*.meta')))):
   rr.append(r)
   if (n+1)%60==0:print('WA10 velocity',n+1,flush=True)
 t,u,s=map(np.stack,zip(*rr));d=dict(time=t,u=u,site=s);np.savez_compressed(dest,**d);dump(tag,dict(signature=sig));return d

def metrics(g,b,new):
 data={};sites={}
 for key in ['C','W','A30','WA30']:
  old={'A30':'A','WA30':'WA'}.get(key,key);d=dict(np.load(OUT/(old+'_extracted.npz')));exp=EXPS[old];assert json.loads((OUT/(old+'_cache.json')).read_text())['signature']==signature(ROOT/'output'/exp);data[key]=d;sites[key]=np.load(OUT/(old+'_site.npz'))['u']
 data['A10']=dict(np.load(STRENGTH/'A10_velocity.npz'));sites['A10']=data['A10']['site'];assert json.loads((STRENGTH/'A10_velocity_sources.json').read_text())['signature']==signature(ROOT/'output'/RUNS['A10']);data['WA10']=new;sites['WA10']=new['site']
 time=pd.DatetimeIndex(new['time']);assert all(np.array_equal(d['time'],new['time']) for d in data.values());use=time>=pd.Timestamp('1996-01-01');tt=time[use];mm=tt.month.to_numpy();phi=b.velocity_mode.values;qw=b.quadrature_weight.values;n=len(qw);rf=g['rf'][:n+1];raw={k:d['u'][use,:n].astype(float) for k,d in data.items()};complete=np.logical_and.reduce([np.isfinite(v).all(axis=(0,1)) for v in raw.values()]);low={};sc={};sl={}
 for k in data:
  raw[k][:,:,~complete]=np.nan;c=np.einsum('zm,tzx,z->tmx',phi,raw[k],qw)/qw.sum();low[k]=np.einsum('zm,tmx->tzx',phi,c);sc[k]=np.einsum('zm,tz,z->tm',phi,sites[k][use,:n],qw)/qw.sum();sl[k]=np.einsum('zm,tm->tz',phi,sc[k])
 rows=[];effects=[];spatial=[];series={};cov=[];boot=[];rng=np.random.default_rng(20261113)
 for kind,fields in [('low',low),('raw_common',raw)]:
  filtered={k:filt(v,mm) for k,v in fields.items()}
  for loc in ['west','central','east','site142']:
   for depth,(lo,hi) in DEPTHS.items():
    dz=np.maximum(0,np.minimum(rf[1:],hi)-np.maximum(rf[:-1],lo));ss={};ff={}
    for k in data:
     if loc=='site142':a=sl[k] if kind=='low' else sites[k][use,:n];ss[k]=a@dz/dz.sum()
     else:
      xlo,xhi=REGIONS[loc];w=dz[:,None]*((g['x']>=xlo)&(g['x']<=xhi))[None];ss[k]=weighted(fields[k],w);sm=np.std(filtered[k][36:-36],axis=0);spatial.append(dict(kind=kind,location=loc,depth=depth,experiment=k,std_rms_cm_s=100*np.sqrt(weighted(sm**2,w))))
     ff[k]=filt(ss[k][:,None],mm)[:,0];series[f'{kind}_{loc}_{depth}_{k}']=ff[k]
    for trim in [24,36,48]:
     f={k:v[trim:-trim] for k,v in ff.items()}
     for k in data:rows.append(dict(kind=kind,location=loc,depth=depth,trim=trim,experiment=k,std_cm_s=100*std(f[k]),mean_cm_s=100*np.mean(ss[k]),ratio_C=std(f[k])/std(f['C']),ratio_W=std(f[k])/std(f['W']),ratio_WA30=std(f[k])/std(f['WA30']),corr_C=corr(f[k],f['C'])))
     for strength in ['10','30']:
      A='A'+strength;WA='WA'+strength;I=f[WA]-f['W']-f[A]+f['C']
      for name,v in [('wind_without_abs',f['W']-f['C']),('wind_with_abs',f[WA]-f[A]),('abs_original_wind',f[A]-f['C']),('abs_bandstop_wind',f[WA]-f['W']),('interaction',I)]:effects.append(dict(kind=kind,location=loc,depth=depth,trim=trim,tau_days=10 if strength=='10' else 30.4375,effect=name,rms_cm_s=100*np.sqrt(np.mean(v*v)),rms_over_C_std=np.sqrt(np.mean(v*v))/std(f['C'])))
    if kind=='low' and loc in ['west','central','site142']:
     f={k:v[36:-36] for k,v in ff.items()};N=len(f['C'])
     for block in [24,36,48]:
      vals=[]
      for _ in range(2000):
       ix=((rng.integers(N,size=(int(np.ceil(N/block)),1))+np.arange(block))%N).ravel()[:N];z={k:v[ix] for k,v in f.items()};I=z['WA10']-z['W']-z['A10']+z['C'];vals.append([std(z['WA10'])/std(z['W']),std(z['WA10'])/std(z['WA30']),np.sqrt(np.mean(I*I))/std(z['C'])])
      ci=np.quantile(vals,[.025,.975],axis=0)
      for j,name in enumerate(['WA10_over_W','WA10_over_WA30','interaction_rms_over_C_std']):boot.append(dict(location=loc,depth=depth,block=block,metric=name,ci_low=ci[0,j],ci_high=ci[1,j]))
 for depth,(lo,hi) in DEPTHS.items():
  dz=np.maximum(0,np.minimum(rf[1:],hi)-np.maximum(rf[:-1],lo));weights=dz@phi/dz.sum()
  for k in data:
   components=filt(sc[k],mm)[36:-36]*weights[None]*100;cv=np.cov(components,rowvar=False,bias=True);total=np.var(components.sum(axis=1));assert abs(cv.sum()-total)<1e-10;cov.append(dict(depth=depth,experiment=k,sum_modal_variance=np.trace(cv),off_diagonal_covariance=cv.sum()-np.trace(cv),total_variance=total))
 df=pd.DataFrame(rows);df.to_csv(DEST/'velocity_metrics.csv',index=False);pd.DataFrame(effects).to_csv(DEST/'factorial_effects.csv',index=False);pd.DataFrame(spatial).to_csv(DEST/'spatial_metrics.csv',index=False);pd.DataFrame(cov).to_csv(DEST/'modal_covariance.csv',index=False);pd.DataFrame(boot).to_csv(DEST/'paired_bootstrap.csv',index=False);pd.DataFrame(series,index=tt).rename_axis('time').to_csv(DEST/'filtered_series.csv')
 # Replay old controls and 10-day results, including covariance identity above.
 old=pd.read_csv(STRENGTH/'velocity_metrics.csv');err=[]
 for _,r in df.query("experiment in ['C','A10']").iterrows():
  v=old.query('kind==@r.kind and location==@r.location and depth==@r.depth and trim==@r.trim and experiment==@r.experiment');err.append(abs(r.std_cm_s-v.std_cm_s.iloc[0]))
 assert max(err)<1e-6;dump(DEST/'validation.json',dict(previous_std_replay_error_cm_s=max(err),complete_profiles=int(complete.sum()),modal_covariance_identity_pass=True,method='1996-2025 anomalies; 4th order zero-phase 2-8 yr; default trim36 =>1999-2022; paired time blocks are not initial-condition ensembles'))
 fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 for ax,loc in zip(axs,['west','central','site142']):
  for k,col in [('W','#0072B2'),('A10','#D55E00'),('WA10','#009E73')]:
   d=df.query("kind=='low' and trim==36 and location==@loc and experiment==@k");ax.plot(range(4),[d.query('depth==@dd').ratio_C.iloc[0] for dd in DEPTHS],'o-',label=k,color=col)
  ax.axhline(1,color='gray');ax.set(title=loc,xticks=range(4),xticklabels=list(DEPTHS),ylabel='Std / CTRL');ax.tick_params(axis='x',rotation=35)
 axs[0].legend();fig.savefig(PLOT/'T01_four_run_depths.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained',sharex=True)
 for r,loc in enumerate(['west','site142']):
  for j,dep in enumerate(['500-1000','2000-3000']):
   for k,col in [('C','black'),('W','#0072B2'),('A10','#D55E00'),('WA10','#009E73')]:axs[r,j].plot(tt[36:-36],series[f'low_{loc}_{dep}_{k}'][36:-36]*100,color=col,label=k)
   axs[r,j].set(title=loc+' '+dep+' m',ylabel='U (cm/s)')
 axs[0,0].legend();fig.savefig(PLOT/'T02_series.png',dpi=180);plt.close(fig)
 print(df.query("kind=='low' and trim==36 and location in ['west','central','site142']").to_string(index=False),flush=True)

def waves(g,b):
 wave.DEST=DEST;setup=wave.prepare(g,b);wave.PAIRS=RUNS;p=ROOT/'output'/RUNS['WA10'];masks={v:np.fromfile(p/f'rbcs_mask_east_{v}.bin',dtype='>f4').reshape(NZ,NY,NX).astype(float) for v in ['TS','U','V']};data={}
 for k in ['W','WA10']:data[k]=wave.extract(k,g,b,setup,masks)
 data['C']=dict(np.load(RETAINED/'research/mechanism/2026-10-20_eastern_absorber_audit/C_wave_components.npz'));data['A10']=dict(np.load(STRENGTH/'A10_wave_components.npz'));t=pd.DatetimeIndex(data['C']['time']);use=t>=pd.Timestamp('1996-01-01');mm=t[use].month.to_numpy();f={k:filt(v['coef'][use],mm)[36:-36] for k,v in data.items()};rows=[]
 for ci,(name,width,nr) in enumerate(wave.CONFIGS):
  gate=(data['C']['lon']>=250)&(data['C']['lon']<255)
  for n in range(5):
   for j,comp in [(0,'K'),(1,'R1')]:
    ss={k:np.mean(v[:,ci,n,j,gate],axis=-1) for k,v in f.items()}
    for k in data:rows.append(dict(config=name,vertical_mode=n+1,component=comp,experiment=k,std=std(ss[k]),ratio_C=std(ss[k])/std(ss['C']),ratio_W=std(ss[k])/std(ss['W']),ratio_A10=std(ss[k])/std(ss['A10'])))
 pd.DataFrame(rows).to_csv(DEST/'wave_components.csv',index=False)
 print('Wave diagnostics complete',flush=True)

def main():
 DEST.mkdir(parents=True,exist_ok=True);PLOT.mkdir(parents=True,exist_ok=True);audit_new();g=grid(ROOT/'output'/RUNS['C']);b=xr.load_dataset(ROOT/'research/model_control/evidence/vertical_mode_basis.nc');new=extract_new(g);metrics(g,b,new);waves(g,b)
if __name__=='__main__':main()
