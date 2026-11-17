"""Native142E transport factorial using existing completed experiments.

Primary: below1900m, coast to2N, long-record1996-2025 2-8year bandpass,
1999-2022. Supplement: original user iafilt on all103observation months.
All six runs, including30day absorption sensitivity, use one native geometry.
No simulations or production configurations are changed.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib,re
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/comparison'))
from compare_channel_transport import model_geometry, integrate, bandpass, meta, fingerprint
from compare_mooring_transport import iafilt,stats
DEST=ROOT/'research/mechanism/2026-11-16_transport_factorial'
PLOT=ROOT/'fig/261116_transport_factorial'
RUNS={'C':'261016_ctrl1','W':'261019_noiav_wind','A10':'261020_eabs_10d','WA10':'261021_noiav_wind_eabs_10d','A30':'261019_eabs','WA30':'261019_noiav_wind_eabs'}
BANDS={'500-1000':(500,1000),'1000-1900':(1000,1900),'1900-3000':(1900,3000),'3000-bottom':(3000,np.inf),'1900-bottom':(1900,np.inf),'1000-3000':(1000,3000),'2000-3000':(2000,3000)}
ALG='native142_transport_factorial_v1'
HASHES={}

def sha(p):
    resolved=p.resolve();key=(str(resolved),resolved.stat().st_size,resolved.stat().st_mtime_ns)
    if key in HASHES:return HASHES[key]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    HASHES[key]=h.hexdigest();return HASHES[key]

def tail(p):
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-30000));return f.read().decode(errors='replace')

def audit():
    records={};names=['mitgcmuv','data','data.cal','data.ggl90','data.obcs','bathy_avg.bin','pickup.clim_r4_1.data','pickup.clim_r4_1.meta','pickup_ggl90.clim_r4_1.data','pickup_ggl90.clim_r4_1.meta','XG.data','YG.data','YC.data','DYG.data','RF.data','RC.data','hFacW.data']
    common=None
    for key,exp in RUNS.items():
        folder=ROOT/'output'/exp;hashes={n:sha(folder/n) for n in names}
        if common is None:common=hashes
        if hashes!=common:raise ValueError(f'Common physical configuration/grid/pickup differs: {key}')
        normal='Execution ended Normally' in tail(folder/'STDOUT.0000')
        if not normal:raise ValueError('Incomplete experiment: '+exp)
        pkg=f90nml.read(folder/'data.pkg')['packages'];exf=f90nml.read(folder/'data.exf');rbcs=f90nml.read(folder/'data.rbcs')['rbcs_parm01']
        if bool(pkg.get('userbcs',False))!=(key in ['A10','WA10','A30','WA30']):raise ValueError('UnexpectedRBCS state')
        records[key]=dict(experiment=exp,normal_end=normal,common_sha256=hashes,uwind=exf['exf_nml_02']['uwindfile'],vwind=exf['exf_nml_02']['vwindfile'],useRBCS=bool(pkg.get('userbcs',False)),tau_seconds={v:rbcs['taurelax'+v] for v in ['t','s','u','v']})
    if records['C']['uwind']!=records['A10']['uwind'] or records['W']['uwind']!=records['WA10']['uwind']:raise ValueError('Unmatched factorial winds')
    (DEST/'source_audit.json').write_text(json.dumps(records,indent=2))
    return records

def extract(key,m):
    folder=ROOT/'output'/RUNS[key];paths=sorted(folder.glob('diag3d.*.meta'));items=[]
    for p in paths:
        nr,fields,t=meta(p)
        if 'UVEL' not in fields:raise ValueError('MissingUVEL')
        if p.with_suffix('.data').stat().st_size!=nr*NZ*NY*NX*4:raise ValueError('Incomplete binary')
        items.append((p,nr,fields,t))
    times=pd.DatetimeIndex([i[3] for i in items]);expected=pd.date_range('1991-01-01',periods=420,freq='MS')
    if not times.equals(expected):raise ValueError('420monthly outputs required')
    files=[p for i in items for p in [i[0],i[0].with_suffix('.data')]]
    sig=fingerprint(files,dict(algorithm=ALG,faces=m['faces'].tolist(),js=m['js'].tolist(),wet_geometry_sha256=sha(folder/'hFacW.data')))
    cache=DEST/f'{key}_source_section.nc';tag=DEST/f'{key}_source_signature.json'
    if cache.exists() and tag.exists() and json.loads(tag.read_text())['signature']==sig:
        with xr.open_dataset(cache) as ds:v=ds.u.values
        return v,times,sig
    def read(item):
        p,nr,fields,t=item;a=np.memmap(p.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,NZ,NY,NX))
        v=np.moveaxis(np.array(a[fields.index('UVEL'),:,m['js'][0]:m['js'][-1]+1,:],float)[:,:,m['faces']],-1,0);v[~m['wet']]=np.nan
        return v
    with ThreadPoolExecutor(max_workers=4) as pool:v=np.stack(list(pool.map(read,items)))
    xr.Dataset(dict(u=(('time','face','depth','latitude'),v)),coords=dict(time=times,face=m['face_longitudes'],depth=m['depth'],latitude=m['lat']),attrs=dict(units='m/s',algorithm=ALG)).to_netcdf(cache)
    tag.write_text(json.dumps(dict(signature=sig,months=len(times)),indent=2))
    return v,times,sig

NZ,NY,NX=40,192,720

def smooth_monthly(frame):
    days=pd.date_range(frame.index[0],frame.index[-1]+pd.offsets.MonthEnd(0),freq='D')
    expanded=frame.reindex(days,method='ffill')
    return pd.DataFrame(iafilt(expanded.values),index=days,columns=frame.columns).resample('MS').mean()

def factorial(frame,period):
    rows=[];effects=[];series={}
    for band in BANDS:
        f=frame.xs(band,level='band',axis=1).droplevel('direction',axis=1)
        for run in RUNS:
            c=f.C;s=f[run]
            rows.append(dict(period=period,band=band,run=run,n=len(f),first=str(f.index[0].date()),last=str(f.index[-1].date()),mean_Sv=float(s.mean()),std_Sv=float(s.std(ddof=0)),std_ratio_to_C=float(s.std(ddof=0)/c.std(ddof=0)),correlation_with_C=float(s.corr(c))))
        for tau in ['10','30']:
            c,w,a,wa=[f[k]-f[k].mean() for k in ['C','W','A'+tau,'WA'+tau]]
            signals={'wind_without_absorber':c-w,'wind_with_absorber':a-wa,'absorber_with_original_wind':c-a,'absorber_with_bandstop_wind':w-wa,'interaction':wa-w-a+c}
            for name,s in signals.items():
                effects.append(dict(period=period,band=band,absorber_days=tau,effect=name,rms_Sv=float(np.sqrt(np.mean(s*s))),rms_over_C_std=float(np.sqrt(np.mean(s*s))/c.std(ddof=0))))
                series[f'{band}__{tau}d__{name}']=s
    return pd.DataFrame(rows),pd.DataFrame(effects),pd.DataFrame(series,index=frame.index)

def main():
    DEST.mkdir(parents=True,exist_ok=True);PLOT.mkdir(parents=True,exist_ok=True)
    audit_info=audit();print('Audit passed: same grid/initial state/physical setup;6normal runs',flush=True)
    m=model_geometry(ROOT/'output'/RUNS['C'],2);weights={}
    for band,(lo,hi) in BANDS.items():
        dz=np.maximum(0,np.minimum(m['bottom'],hi)-np.maximum(m['top'],lo))
        weights[band]=dz*(m['metric_per_degree']*(m['north']-m['south']))[:,None,:]*m['face_weights'][:,None,None]
    table={};latitude={};checks={};signatures={}
    for key in RUNS:
        print('Extracting',key,RUNS[key],flush=True);v,times,sig=extract(key,m);signatures[key]=sig
        for band,area in weights.items():
            for direction,q in zip(['net','east','west'],integrate(v,area)):table[(key,band,direction)]=q
            lat=np.where(area[None]>0,v*area[None],0).sum(axis=(1,2))/1e6;latitude[(key,band)]=lat
            if not np.allclose(lat.sum(axis=1),table[(key,band,'net')],atol=1e-10):raise AssertionError('Latitude flux closure')
        if not np.allclose(table[(key,'1900-bottom','net')],table[(key,'1900-3000','net')]+table[(key,'3000-bottom','net')],atol=1e-10):raise AssertionError('Depth partition closure')
        # Independent source summation at three dates, bypassing cached section.
        raw_errors=[]
        for index in [0,210,419]:
            p=sorted((ROOT/'output'/RUNS[key]).glob('diag3d.*.meta'))[index];nr,fields,_=meta(p)
            source=np.memmap(p.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,NZ,NY,NX));q=0.;area=weights['1900-bottom']
            for f,face in enumerate(m['faces']):
                z=np.asarray(source[fields.index('UVEL'),:,m['js'],face],float)
                if z.shape!=area[f].shape:z=z.T
                mask=area[f]>0;q+=float(np.sum(z[mask]*area[f][mask]))/1e6
            raw_errors.append(abs(q-table[(key,'1900-bottom','net')][index]))
        if max(raw_errors)>1e-10:raise AssertionError('Independent source flux error')
        checks[key]=dict(source_months=420,source_signature=sig,independent_months=3,independent_max_error_Sv=max(raw_errors),direction_and_depth_and_latitude_closure=True)
        print('Integrated',key,flush=True)
    raw=pd.DataFrame(table,index=times);raw.columns=pd.MultiIndex.from_tuples(raw.columns,names=['run','band','direction']);raw.to_csv(DEST/'raw_transports_420months.csv',index_label='time')
    net=raw.xs('net',level='direction',axis=1,drop_level=False)
    recent=net.loc['2015-12':'2024-06'];annual=smooth_monthly(recent)
    longraw=net.loc['1996-01':'2025-12'];longannual=smooth_monthly(longraw)
    strict=pd.DataFrame({k:bandpass(v.values,longraw.index.month.to_numpy()) for k,v in longraw.items()},index=longraw.index);strict.columns=longraw.columns;strictmain=strict.iloc[36:-36]
    for name,f in [('raw_recent',recent),('iafilt103',annual),('iafilt_long1996_2025',longannual),('bandpass1999_2022',strictmain)]:f.to_csv(DEST/(name+'.csv'),index_label='time')
    # Use long-record support, then subset the same103months to diagnose the
    # temporal support effect separately from the primary matched-observation method.
    longsupport_recent=longannual.loc['2015-12':'2024-06']
    metrics=[];effects=[]
    for name,f in [('raw_recent',recent),('iafilt103',annual),('iafilt_long1996_2025',longannual),('iafilt_longsupport_recent103',longsupport_recent),('bandpass1999_2022',strictmain)]:
        mm,ee,dd=factorial(f,name);metrics.append(mm);effects.append(ee);dd.to_csv(DEST/(name+'_factorial_effect_series.csv'),index_label='time')
    metrics=pd.concat(metrics,ignore_index=True);effects=pd.concat(effects,ignore_index=True)
    metrics.to_csv(DEST/'metrics.csv',index=False);effects.to_csv(DEST/'factorial_effect_metrics.csv',index=False)
    # Check C reproduces the previously audited native flux and smoother.
    old=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/261016_ctrl1/transport/transports.csv',index_col=0,parse_dates=True)
    q=raw[('C','1900-bottom','net')];common=q.index.intersection(old.index)
    baseline_error=float(abs(q.loc[common]-old.loc[common,'model_native_net_Sv']).max())
    prior=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/monthly_smoothed_Sv.csv',index_col=0,parse_dates=True)
    smooth_error=float(abs(annual[('C','1900-bottom','net')]-prior.ctrl1).max())
    if max(baseline_error,smooth_error)>1e-10:raise AssertionError('Existing CTRL reproduction')
    # Relative uncertainty scale from existing five-member CTRL; it is a
    # descriptive benchmark, not matched perturbed ensembles for all treatments.
    ens=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/monthly_smoothed_Sv.csv',index_col=0,parse_dates=True)
    spread_rms=float(np.sqrt(np.mean(ens.member_std_ddof0.values**2)))
    observed=ens.obs;reference=ens.glorys;comparison=[]
    for key in RUNS:
        for name,ref in [('obs',observed),('glorys',reference)]:comparison.append(dict(run=key,reference=name,**stats(annual[(key,'1900-bottom','net')],ref)))
    pd.DataFrame(comparison).to_csv(DEST/'recent_reference_comparison.csv',index=False)
    # Latitude diagnostics retained before filtering to reveal cancellation.
    xr.Dataset({f'{key}_{band.replace("-","_")}':(('time','latitude'),value) for (key,band),value in latitude.items()},coords=dict(time=times,latitude=m['lat']),attrs=dict(units='Sv',definition='Native weighted Uface flux per latitude cell')).to_netcdf(DEST/'latitude_transports.nc')
    geom=dict(face_longitudes=m['face_longitudes'].tolist(),face_weights=m['face_weights'].tolist(),coast_latitude=m['coast'],north=2,band_areas_km2={k:float(v.sum()/1e6) for k,v in weights.items()})
    (DEST/'geometry.json').write_text(json.dumps(geom,indent=2))
    (DEST/'validation.json').write_text(json.dumps(dict(per_run=checks,baseline_transport_error_Sv=baseline_error,baseline_iafilt_error_Sv=smooth_error,all_same_native_geometry=True,all_recent_metrics103months=True,initial_spread103_RMS_Sv=spread_rms),indent=2))
    fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    colors={'C':'black','W':'#009E73','A10':'#D55E00','WA10':'#CC79A7','A30':'#E69F00','WA30':'#56B4E9'}
    for key in RUNS:
        axs[0].plot(recent.index,recent[(key,'1900-bottom','net')],label=key,color=colors[key],lw=1.2,ls='--' if key.endswith('30') else '-')
        axs[1].plot(annual.index,annual[(key,'1900-bottom','net')],label=key,color=colors[key],lw=1.6,ls='--' if key.endswith('30') else '-')
    for ax in axs:ax.axhline(0,color='gray',lw=.6);ax.set_ylabel('Eastward transport (Sv)');ax.legend(ncol=6)
    axs[0].set_title('Raw monthly channel transport |142E below1900m');axs[1].set_title('Original iafilt | all103months,2015-12 to2024-06 | endpoints retained')
    fig.savefig(PLOT/'QF01_factorial_transport.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,period,title in zip(axs,['iafilt103','bandpass1999_2022'],['365day smoothing:103months','2-8year anomalies:1999-2022']):
        f=metrics.query('period==@period').pivot(index='band',columns='run',values='std_Sv').loc[list(BANDS)]
        x=np.arange(len(f));width=.19
        for j,key in enumerate(['C','W','A10','WA10']):ax.bar(x+(j-1.5)*width,f[key],width,label=key,color=colors[key])
        ax.set_xticks(x,f.index,rotation=40,ha='right');ax.set(ylabel='Transport std (Sv)',title=title);ax.legend(ncol=2)
    fig.savefig(PLOT/'QF02_depth_bands.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for ax,period,title in zip(axs,['iafilt103','bandpass1999_2022'],['Recent103month original iafilt','Long-record2-8years,1999-2022']):
        ff=annual if period=='iafilt103' else strictmain
        a={k:ff[(k,'1900-bottom','net')]-ff[(k,'1900-bottom','net')].mean() for k in RUNS}
        for name,v in [('C-W',a['C']-a['W']),('C-A10',a['C']-a['A10']),('W-WA10',a['W']-a['WA10']),('Interaction',a['WA10']-a['W']-a['A10']+a['C'])]:ax.plot(v.index,v,label=name,lw=1.2)
        ax.axhline(0,color='gray',lw=.5);ax.set(title=title,ylabel='Paired anomaly difference (Sv)');ax.legend(ncol=4)
    fig.savefig(PLOT/'QF03_factorial_anomaly_differences.png',dpi=180);plt.close(fig)
    info=dict(algorithm=ALG,runs=RUNS,geometry=geom,supplementary_observation_period=dict(period='2015-12 to2024-06',n=103,filter='Original iafilt,365sample cycle,mean restoration,365day reflect,all endpoints kept'),primary=dict(period='1999-2022',support='1996-2025',filter='2-8year fourthorder zerophase Butterworth,monthly climate removed,36months trimmed'),initial_condition_spread_rms_Sv=spread_rms,main_metrics=metrics.query("period=='bandpass1999_2022' and band=='1900-bottom'").to_dict('records'),main_effects=effects.query("period=='bandpass1999_2022' and band=='1900-bottom'").to_dict('records'),limits=['Perturbations are interventions, not independent contributions','RBCS dampensTSUV and is not proven reflectionoff','Wind speed changes bulk stress and heatflux','Initial-condition benchmark fromCTRL only, not an ensemble for each intervention','Formal long-record mechanism basis remains distinct from observation-period validation'])
    (DEST/'summary.json').write_text(json.dumps(info,indent=2));(DEST/'manifest.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),source_signatures=signatures),indent=2))
    print(metrics.query("band=='1900-bottom'").to_string(index=False),flush=True)
    print(effects.query("band=='1900-bottom'").to_string(index=False),flush=True)

if __name__=='__main__':main()
