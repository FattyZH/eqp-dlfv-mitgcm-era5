"""Assess western deep U proxies; these are not section-integrated transports."""
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from mitkit.paths import project_root

ROOT=project_root()
EXPS=['260821_ctrl','260916_ctrl_kpp']
DEST=ROOT/'research/deep_transport_assessment'


def bandpass(a, trim):
    a=a.groupby('time.month')-a.groupby('time.month').mean('time')
    x=a.values; good=np.isfinite(x).all(axis=0)
    out=np.full(x.shape,np.nan)
    out[:,good]=sosfiltfilt(butter(4,[1/8,1/2],fs=12,btype='bandpass',output='sos'),x[:,good],axis=0)
    return xr.DataArray(out,coords=a.coords,dims=a.dims).isel(time=slice(trim,-trim))


def weights(a, lo, hi):
    z=a.depth.values; edges=np.r_[0,(z[:-1]+z[1:])/2,z[-1]]
    return xr.DataArray(np.maximum(0,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo)),dims='depth',coords={'depth':a.depth})


def region(a,lo=1000,hi=3000):
    return a.sel(lon=slice(140,170)).weighted(weights(a,lo,hi)).mean(('depth','lon'))


def metrics(m,c):
    return dict(r=float(np.corrcoef(m,c)[0,1]),std_ratio=float(np.std(m)/np.std(c)),model_std_cm_s=float(np.std(m)*100),cmems_std_cm_s=float(np.std(c)*100),rmse_cm_s=float(np.sqrt(np.mean((m-c)**2))*100),nrmse=float(np.sqrt(np.mean((m-c)**2))/np.std(c)),skill_vs_zero_anomaly=float(1-np.mean((m-c)**2)/np.mean(c**2)))


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    rows=[]; series={}; local=[]
    for exp in EXPS:
        ds=xr.load_dataset(ROOT/'fig'/f'{exp}_cmems_comparison/sections.nc')
        for trim in [24,36,48]:
            m,c=bandpass(ds.model,trim),bandpass(ds.cmems,trim)
            for lo,hi in [(1000,3000),(1000,1500),(1500,2000),(2000,3000)]:
                a,b=region(m,lo,hi),region(c,lo,hi)
                row=dict(experiment=exp,trim=trim,depth=f'{lo}-{hi}',**metrics(a.values,b.values))
                lags=np.arange(-12,13)
                rr=[np.corrcoef(a.values[k:],b.values[:-k])[0,1] if k>0 else np.corrcoef(a.values[:k],b.values[-k:])[0,1] if k<0 else row['r'] for k in lags]
                row.update(best_lag_months=int(lags[np.argmax(rr)]),best_lag_r=float(np.max(rr)))
                rows.append(row)
                if trim==36 and lo==1000 and hi==3000:
                    series[exp]=(a.values,b.values)
                    for label,idx in [('first_half',slice(0,len(a)//2)),('second_half',slice(len(a)//2,None))]:
                        rows.append(dict(experiment=exp,trim=36,depth='1000-3000',subset=label,**metrics(a.values[idx],b.values[idx])))
            if trim==36:
                corr=xr.corr(m,c,dim='time').sel(lon=slice(140,170),depth=slice(1000,3000))
                ratio=(m.std('time')/c.std('time')).sel(lon=slice(140,170),depth=slice(1000,3000))
                local.append(dict(experiment=exp,point_r_quantiles=np.nanquantile(corr,[.1,.5,.9]).tolist(),point_std_ratio_quantiles=np.nanquantile(ratio,[.1,.5,.9]).tolist(),fraction_points_r_below_0_5=float((corr<.5).sum()/corr.notnull().sum()),deep_west_mean_bias_cm_s=float(region(ds.model-ds.cmems).mean()*100)))
    pd.DataFrame(rows).to_csv(DEST/'western_deep_metrics.csv',index=False)
    # Paired circular moving-block bootstrap; an exploratory sampling interval,
    # not a representation of reanalysis uncertainty or mechanism attribution.
    rng=np.random.default_rng(20261008); n=len(series[EXPS[0]][0]); boot={e:[] for e in EXPS}; deltas=[]
    for _ in range(1000):
        starts=rng.integers(0,n,size=int(np.ceil(n/36)))
        ids=((starts[:,None]+np.arange(36))%n).ravel()[:n]
        for e in EXPS:
            a,b=series[e]; boot[e].append(metrics(a[ids],b[ids]))
        deltas.append(boot[EXPS[1]][-1]['r']-boot[EXPS[0]][-1]['r'])
    ci={e:{k:np.quantile([v[k] for v in boot[e]],[.025,.975]).tolist() for k in ['r','std_ratio','rmse_cm_s']} for e in EXPS}
    result=dict(domain='140-170 E; equatorial latitude point average -0.5 to 0.5; thickness-weighted 1000-3000 m U proxy',local=local,bootstrap_95_percent=ci,paired_kpp_minus_ctrl_r_interval=np.quantile(deltas,[.025,.975]).tolist(),bootstrap='1000 paired circular moving-block resamples, 36-month blocks; conditional on filtered data; exploratory, not observation uncertainty',lag_definition='positive k: correlate model(t+k) with CMEMS(t), so model lags CMEMS',warning='Not transport in Sv; cell-area and partial-cell integration are required for a physical section transport.')
    (DEST/'assessment.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    print(pd.DataFrame(rows).query("trim == 36").to_string(index=False))

if __name__=='__main__': main()
