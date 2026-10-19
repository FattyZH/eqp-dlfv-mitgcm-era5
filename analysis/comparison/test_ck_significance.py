"""Paired moving-block bootstrap and actual GGL90 coefficient comparison."""
import re
import argparse
import f90nml
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from assess_western_deep import ROOT, DEST, bandpass, weights
EXPS=['261008_ck_0p07','261008_ck_0p10','261008_ck_0p14']
OUT=ROOT/'fig/ck_sensitivity_suite'
BASE_INDEX=1
LOW_DIR=None
LAYERS=[(1000,3000),(1000,1500),(1500,2000),(2000,3000)]


def stats(a,c):
    ac=a-a.mean(-1,keepdims=True);cc=c-c.mean(-1,keepdims=True)
    return np.stack([(ac*cc).sum(-1)/np.sqrt((ac*ac).sum(-1)*(cc*cc).sum(-1)),a.std(-1)/c.std(-1),np.sqrt(((a-c)**2).mean(-1))*100],axis=-1)


def bootstrap():
    rng=np.random.default_rng(20261009);rows=[]
    for location in ['142E0','130-150E']:
        if LOW_DIR is not None:
            ref=xr.load_dataset(LOW_DIR/'CMEMS_reconstruction.nc')
            ds=[xr.Dataset({'model':xr.load_dataset(LOW_DIR/f'{e}_reconstruction.nc').low_u,'cmems':ref.low_u}) for e in EXPS]
            ds=[d.interp(lon=142).drop_vars('lon') if location=='142E0' else d.sel(lon=slice(130,150)) for d in ds]
        elif location=='142E0': ds=[xr.load_dataset(DEST/f'{e}_site_142E_0N.nc') for e in EXPS]
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
                for i in range(len(EXPS)):
                    if i==BASE_INDEX:continue
                    for j,key in enumerate(['correlation','std_ratio','anomaly_rmse_cm_s']):
                        delta=vals[i,:,j]-vals[BASE_INDEX,:,j];l,u=np.quantile(delta,[.025,.975])
                        rows.append(dict(location=('142E_equatorial_band' if LOW_DIR is not None and location=='142E0' else location),depth=f'{lo}-{hi}',experiment=EXPS[i],baseline=EXPS[BASE_INDEX],block_months=block,metric=key,delta=float(obs[i,j]-obs[BASE_INDEX,j]),ci_low=l,ci_high=u,excludes_zero=bool(l>0 or u<0)))
                    delta=np.abs(bias[i][bid].mean(-1))-np.abs(bias[BASE_INDEX][bid].mean(-1));l,u=np.quantile(delta,[.025,.975])
                    rows.append(dict(location=('142E_equatorial_band' if LOW_DIR is not None and location=='142E0' else location),depth=f'{lo}-{hi}',experiment=EXPS[i],baseline=EXPS[BASE_INDEX],block_months=block,metric='absolute_mean_bias_cm_s',delta=abs(bias[i].mean())-abs(bias[BASE_INDEX].mean()),ci_low=l,ci_high=u,excludes_zero=bool(l>0 or u<0)))
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
        parms=f90nml.read(p/'data')['parm01']
        floors=[float(parms.get('viscar',0)),float(parms.get('diffkrs',0))]
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
                    accum.setdefault(k,[]).append((vv.mean(),np.mean(np.abs(vv-floors[f])<=max(floors[f]*.01,1e-12))))
            del a
        for (key,lo,hi),vals in accum.items():
            vals=np.asarray(vals)
            rows.append(dict(experiment=e,field=key,depth=f'{lo}-{hi}',background_floor_m2_s=floors[0 if key=='GGL90ArU' else 1],months=len(vals),mean_m2_s=vals[:,0].mean(),fraction_monthly_means_near_floor=vals[:,1].mean()))
    pd.DataFrame(rows).to_csv(OUT/'actual_mixing_coefficients.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp',nargs='+',default=EXPS)
    parser.add_argument('--baseline',default='261008_ck_0p10')
    parser.add_argument('--out',default='fig/ck_sensitivity_suite')
    parser.add_argument('--low-dir',default=None,help='Use low-mode reconstructions instead of raw fields')
    parser.add_argument('--skip-mixing',action='store_true')
    parser.add_argument('--mixing-only',action='store_true')
    args=parser.parse_args()
    EXPS=args.exp;BASE_INDEX=EXPS.index(args.baseline);OUT=ROOT/args.out
    LOW_DIR=ROOT/args.low_dir if args.low_dir else None
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.mixing_only:bootstrap();print('Bootstrap complete',flush=True)
    if not args.skip_mixing:mixing();print('Mixing coefficients complete',flush=True)
