"""Shared read-only MDS, weighting and filtering helpers for active analyses.

Extracted unchanged from the retained four-run analysis on 2026-11-13.
Historical result paths point into archive/retained_2026-11-13.
"""
from pathlib import Path
import argparse, hashlib, json, re, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
import f90nml
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RETAINED=ROOT/'archive/retained_2026-11-13'
OUT=RETAINED/'research/mechanism/2026-10-20_four_experiments'
FIG=RETAINED/'fig/261020_mechanism_four'
EXPS={'C':'261016_ctrl','W':'261019_noiav_wind','A':'261019_eabs','WA':'261019_noiav_wind_eabs'}
REGIONS={'west':(130,150),'central':(170,220),'east':(255,280)}
BANDS={'upper':(0,300),'middle':(1000,2000),'deep':(1000,3000),'deepest':(2000,3000)}
SURF=['ETAN','PHIBOT','oceTAUX','oceTAUY','MXLDEPTH','TFLUX','oceFWflx']
MOM=['TOTUTEND','Um_dPhiX','Um_Cori','Um_Advec','Um_Diss','Um_ImplD','Um_Ext','Um_Metr','Um_hDis4']
NX,NY,NZ=720,192,40

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def tail(p):
 with open(p,'rb') as f:f.seek(max(0,p.stat().st_size-20000));return f.read().decode(errors='replace')

def meta(p):
 s=p.read_text(); fields=re.findall(r"'([^']+)'",s.split('fldList')[1]) if 'fldList' in s else []
 n=int(re.search(r'nrecords\s*=\s*\[\s*(\d+)',s)[1]);date=re.search(r"timeStepDate\s*=\s*\[\s*'([^']+)'",s)[1]
 interval=[float(x) for x in re.search(r'timeInterval\s*=\s*\[([^]]+)',s)[1].split()]
 dt=pd.Timestamp(date).tz_localize(None)-pd.Timedelta(seconds=(interval[1]-interval[0])/2)
 return n,[f.strip() for f in fields],dt.to_period('M').to_timestamp()

def grid(p):
 def read(n,shape):return np.fromfile(p/(n+'.data'),dtype='>f4').reshape(shape).astype(float)
 return dict(x=read('XG',(NY,NX))[0],xc=read('XC',(NY,NX))[0],y=read('YC',(NY,NX))[:,0],z=-read('RC',(NZ,)),rf=-read('RF',(NZ+1,)),hw=read('hFacW',(NZ,NY,NX)),hc=read('hFacC',(NZ,NY,NX)))

def signature(p):
 fs=[p/f for f in ['XG.data','YC.data','RF.data','hFacW.data','hFacC.data','data.diagnostics']]
 for pref in ['diag3d','diag2d','diag_mom_u']:fs+=sorted(p.glob(pref+'.*.*'))
 return hashlib.sha256(json.dumps([(q.name,q.stat().st_size,q.stat().st_mtime_ns) for q in fs]).encode()).hexdigest()

def filt(a,months):
 a=np.array(a,float);valid=np.isfinite(a).all(axis=0);out=np.full(a.shape,np.nan);x=a[:,valid]
 for m in range(1,13):x[months==m]-=np.mean(x[months==m],axis=0)
 out[:,valid]=sosfiltfilt(butter(4,[1/8,1/2],fs=12,btype='bandpass',output='sos'),x,axis=0)
 return out

def std(a):return float(np.std(a))

def corr(a,b):return float(np.corrcoef(a,b)[0,1])

def weighted(a,w):
 rr,cc=np.nonzero(w>0);rs=slice(rr.min(),rr.max()+1);cs=slice(cc.min(),cc.max()+1)
 a=a[...,rs,cs];w=w[rs,cs]
 return np.nansum(a*w,axis=(-2,-1))/np.sum(np.isfinite(a)*w,axis=(-2,-1))
