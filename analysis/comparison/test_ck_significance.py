"""Paired moving-block bootstrap and actual GGL90 coefficient comparison."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from assess_western_deep import ROOT, DEST, bandpass, weights
EXPS=['261008_ck_0p07','261008_ck_0p10','261008_ck_0p14']
OUT=ROOT/'fig/ck_sensitivity_suite'
LAYERS=[(1000,3000),(1000,1500),(1500,2000),(2000,3000)]


def stats(a,c):
    ac=a-a.mean(-1,keepdims=True);cc=c-c.mean(-1,keepdims=True)
    return np.stack([(ac*cc).sum(-1)/np.sqrt((ac*ac).sum(-1)*(cc*cc).sum(-1)),a.std(-1)/c.std(-1),np.sqrt(((a-c)**2).mean(-1))*100],axis=-1)


def bootstrap():
    rng=np.random.default_rng(20261009);rows=[]
    for location in ['142E0','130-150E']:
        if location=='142E0': ds=[xr.load_dataset(DEST/f'{e}_site_142E_0N.nc') for e in EXPS]
        else: ds=[xr.load_dataset(ROOT/'fig'/f'{e}_cmems_comparison/sections.nc').sel(lon=slice(130,150)) for e in EXPS]
        ds=xr.align(*ds,join='inner');good=ds[0].cmems.notnull().all('time')
        for d in ds:good=good&d.model.notnull().all('time')
        for lo,hi in LAYERS:
            def avg(a):
                a=a.where(good)
                return a.weighted(weights(a,lo,hi)).mean(('depth','lon') if 'lon' in a.dims else 'depth')
            # Linear averaging and filtering commute for the common fixed mask.
            a=np.stack([bandpass(avg(d.model),36).values for d in ds]);c=bandpass(avg(ds[0].cmems),36).values
            bias=np.stack([avg(d.model-d.cmems).values*100 for d in ds])
            n=a.shape[-1];obs=stats(a,c)
            for block in [24,36,48]:
                starts=rng.integers(n,size=(3000,int(np.ceil(n/block))))
                ids=((starts[...,None]+np.arange(block))%n).reshape(3000,-1)[:,:n]
                vals=np.stack([stats(v[ids],c[ids]) for v in a])
                # Mean bias uses full unfiltered common period, with matched blocks.
                nb=bias.shape[-1];st=rng.integers(nb,size=(3000,int(np.ceil(nb/block))))
                bid=((st[...,None]+np.arange(block))%nb).reshape(3000,-1)[:,:nb]
                for i in [0,2]:
                    for j,key in enumerate(['correlation','std_ratio','anomaly_rmse_cm_s']):
                        delta=vals[i,:,j]-vals[1,:,j];l,u=np.quantile(delta,[.025,.975])
                        rows.append(dict(location=location,depth=f'{lo}-{hi}',experiment=EXPS[i],baseline=EXPS[1],block_months=block,metric=key,delta=float(obs[i,j]-obs[1,j]),ci_low=l,ci_high=u,excludes_zero=bool(l>0 or u<0)))
                    delta=np.abs(bias[i][bid].mean(-1))-np.abs(bias[1][bid].mean(-1));l,u=np.quantile(delta,[.025,.975])
                    rows.append(dict(location=location,depth=f'{lo}-{hi}',experiment=EXPS[i],baseline=EXPS[1],block_months=block,metric='absolute_mean_bias_cm_s',delta=abs(bias[i].mean())-abs(bias[1].mean()),ci_low=l,ci_high=u,excludes_zero=bool(l>0 or u<0)))
    pd.DataFrame(rows).to_csv(OUT/'bootstrap_differences.csv',index=False)


def mixing():
    rows=[]
    for e in EXPS:
        p=ROOT/'output'/e
        def read(k,shape):return np.fromfile(p/f'{k}.data',dtype='>f4').reshape(shape)
        x=read('XG',(192,720))[0];y=read('YC',(192,720))[:,0]
        js=np.flatnonzero(abs(y)<=.5);ii=np.flatnonzero((x>=130)&(x<=150))
        z=-read('RF',(41,))[:-1]
        h=read('hFacW',(40,192,720))[:,js][:,:,ii]>0
        hc=read('hFacC',(40,192,720))[:,js][:,:,ii]>0
        masks=[h&np.roll(h,1,axis=0),hc&np.roll(hc,1,axis=0)]
        accum={}
        for meta in sorted(p.glob('diag_mix_ggl90.*.meta')):
            txt=meta.read_text();date=re.search(r"timeStepDate\s*=\s*\[\s*'([^']+)'",txt).group(1)
            month=(pd.Timestamp(date)-pd.Timedelta(seconds=2629800/2)).tz_localize(None).to_period('M')
            if month<pd.Period('1996-07') or month>pd.Period('2024-06'):continue
            a=np.memmap(meta.with_suffix('.data'),dtype='>f4',mode='r',shape=(4,40,192,720))
            for f,key in enumerate(['GGL90ArU','GGL90Kr']):
                v=np.asarray(a[f][:,js][:,:,ii],dtype=float)
                for lo,hi in LAYERS:
                    mask=masks[f]&((z>=lo)&(z<hi))[:,None,None]
                    vv=v[mask];k=(key,lo,hi)
                    accum.setdefault(k,[]).append((vv.mean(),np.mean(vv<=1e-5*1.01)))
            del a
        for (key,lo,hi),vals in accum.items():
            vals=np.asarray(vals)
            rows.append(dict(experiment=e,field=key,depth=f'{lo}-{hi}',months=len(vals),mean_m2_s=vals[:,0].mean(),fraction_monthly_means_near_floor=vals[:,1].mean()))
    pd.DataFrame(rows).to_csv(OUT/'actual_mixing_coefficients.csv',index=False)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    bootstrap();print('Bootstrap complete',flush=True)
    mixing();print('Mixing coefficients complete',flush=True)
