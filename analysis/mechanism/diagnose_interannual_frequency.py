"""CMEMS 2–8 year depth-limited CEOF and deep-transport spectra.

Frequency selection follows CMEMS; CTRL is only a comparison. No unfiltered
CEOF, no lag fitting, no zero-padding or nominal monthly-sample significance.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import hilbert,periodogram
from scipy.signal.windows import dpss
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compute_ctrl_ceof import decompose
from diagnose_depth_amplitude import filt

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/mechanism/2026-11-16_depth_amplitude'
OUT=ROOT/'research/mechanism/2026-11-19_interannual_frequency'
FIG=ROOT/'fig/261119_interannual_frequency'
SCOPES={'basin_500_3000':(130,285,500,3000),'basin_1000_3000':(130,285,1000,3000),'west_500_3000':(130,150,500,3000),'west_1000_3000':(130,150,1000,3000)}

def psd(y,method,weights=None):
    """For complex TCs use P(real)+P(imag), invariant to CEOF phase gauge."""
    y=y-y.mean(axis=0)
    windows=['hann'] if method=='Hann' else list(dpss(len(y),1.5,Kmax=2,sym=False,norm=2))
    estimates=[]
    for window in windows:
        f,p=periodogram(y.real,fs=12,window=window,detrend=False,axis=0,scaling='density')
        if np.iscomplexobj(y):
            _,pi=periodogram(y.imag,fs=12,window=window,detrend=False,axis=0,scaling='density')
            p+=pi
        if weights is not None:p=p@weights/weights.sum()
        estimates.append(p)
    return f,np.mean(estimates,axis=0)

def bandpower(f,p,lo,hi):
    ff=np.r_[lo,f[(f>lo)&(f<hi)],hi]
    return float(trapezoid(np.interp(ff,f,p),ff))

def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'source_sections.npz') as a:
        gl=a['gl12'];ctrl=a['model'];time=pd.DatetimeIndex(a['time']);kk=a['kk']
    with xr.open_dataset(SOURCE/'section_statistics.nc') as d:x=d.longitude.values;z=d.depth.values
    common=np.isfinite(gl).all(axis=0)&np.isfinite(ctrl).all(axis=0)
    filtered=filt(gl,time.month.to_numpy())
    analytic=np.full(filtered.shape,np.nan+1j*np.nan)
    analytic[:,common]=hilbert(filtered[:,common],axis=0)
    analytic=analytic[36:-36];dates=time[36:-36]
    rf=-np.fromfile(ROOT/'output/261016_ctrl1/RF.data',dtype='>f4').astype(float)
    spectra={};curves=[];summary=[];variance=[];checks={};tcseries={'time':dates};bandrows=[]
    def record(label,y,unit,weights=None):
        for method in ['Hann','DPSS_NW1.5_K2']:
            f,p=psd(y,method,weights)
            spectra[(label,method)]=(f,p)
            take=(f>=1/8)&(f<=1/2)
            peak=float(f[take][np.argmax(p[take])]);total=bandpower(f,p,1/8,1/2)
            summary.append(dict(signal=label,method=method,peak_frequency_per_year=peak,peak_period_years=1/peak,power_2_8=total,units=unit,fraction_2_4=bandpower(f,p,1/4,1/2)/total,fraction_4_8=bandpower(f,p,1/8,1/4)/total))
            for lo,hi in [(2,3),(3,5),(5,8),(2,4),(4,8)]:bandrows.append(dict(signal=label,method=method,period_min_years=lo,period_max_years=hi,power_fraction_within_2_8=bandpower(f,p,1/hi,1/lo)/total))
            curves.extend(dict(signal=label,method=method,frequency_per_year=float(ff),PSD=float(pp),units=unit) for ff,pp in zip(f,p))
    for scope,(lo,hi,top,bot) in SCOPES.items():
        dz=np.maximum(0,np.minimum(rf[kk+1],bot)-np.maximum(rf[kk],top))
        weights=np.broadcast_to(dz[:,None],common.shape)
        mask=common&(weights>0)&((x>=lo)&(x<=hi))[None]
        sm,tc,fractions,check=decompose(analytic,mask,weights)
        checks[scope]={**check,'points':int(mask.sum()),'depth_centers_used_m':z[mask.any(axis=1)].tolist(),'nonzero_clipped_layer_thickness_m':dz[mask.any(axis=1)].tolist()}
        np.savez_compressed(OUT/f'GL12_{scope}_ceof.npz',spatial_mode=sm,time_coefficient=tc,explained_variance=fractions,mask=mask,weights=weights,time=dates.values,longitude=x,depth=z)
        for i in range(4):
            variance.append(dict(scope=scope,mode=i+1,explained_variance=float(fractions[i]),cumulative_variance=float(fractions[:i+1].sum())))
            record(f'GL12_{scope}_CEOF{i+1}',tc[:,i],'TC variance / (cycle/year)')
            tcseries[f'{scope}_CEOF{i+1}_real']=tc[:,i].real;tcseries[f'{scope}_CEOF{i+1}_imag']=tc[:,i].imag
        record(f'GL12_{scope}_total_U',filtered[36:-36,mask],'(m/s)^2 / (cycle/year)',weights[mask])
        # Independent phase-gauge check for spectral normalization.
        f,p=psd(tc[:,0],'Hann');_,rot=psd(tc[:,0]*np.exp(.73j),'Hann')
        assert np.allclose(p,rot,rtol=1e-12,atol=1e-12)
        checks[scope]['phase_rotation_psd_max_error']=float(abs(p-rot).max())
        print(scope,'CEOF1-4 %',np.round(fractions[:4]*100,2),flush=True)
    for key in ['GL12','CTRL']:
        with np.load(ROOT/f'research/mechanism/2026-11-17_ctrl_gl12_ceof/{key}_ceof_RF_common.npz') as a:
            assert pd.DatetimeIndex(a['time']).equals(dates)
            for i in [0,1]:record(f'{key}_full_CEOF{i+1}',a['time_coefficient'][:,i],'TC variance / (cycle/year)')
    q=pd.read_csv(ROOT/'research/mechanism/2026-11-17_historical_bandpass/transport_series.csv',parse_dates=['time'])
    q=q.set_index('time').loc[dates]
    for key in ['GL12','CTRL']:
        values=q[f'1900-bottom_{key}_common_bp28'].to_numpy()
        assert np.isfinite(values).all()
        record(f'{key}_Q_below1900_common',values,'Sv^2 / (cycle/year)')
    pd.DataFrame(summary).to_csv(OUT/'spectral_summary.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'spectral_curves.csv',index=False)
    pd.DataFrame(variance).to_csv(OUT/'deep_ceof_variance.csv',index=False)
    pd.DataFrame(bandrows).to_csv(OUT/'candidate_band_power.csv',index=False)
    pd.DataFrame(tcseries).to_csv(OUT/'deep_ceof_time_coefficients.csv',index=False)
    # Candidate selection is based on CMEMS; normalize curves only for shape.
    panels=[('CMEMS: basin-wide CEOF1',['GL12_full_CEOF1','GL12_basin_500_3000_CEOF1','GL12_basin_1000_3000_CEOF1']),('CMEMS: western Pacific CEOF1',['GL12_west_500_3000_CEOF1','GL12_west_1000_3000_CEOF1']),('CMEMS: total spatially weighted U spectrum',[f'GL12_{k}_total_U' for k in SCOPES]),('142°E transport below 1900 m',['GL12_Q_below1900_common','CTRL_Q_below1900_common'])]
    for method,tag in [('Hann','Hann'),('DPSS_NW1.5_K2','DPSS')]:
        fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for ax,(title,labels) in zip(axs.flat,panels):
            for label in labels:
                f,p=spectra[(label,method)];norm=bandpower(f,p,1/8,1/2)
                display=label.replace('GL12_','').replace('_CEOF1','').replace('_total_U','')
                if 'Q_below1900' in label:display='CMEMS Q' if label.startswith('GL12') else 'CTRL Q'
                ax.plot(f,p/norm,'o-',ms=3,label=display)
            ax.axvspan(1/8,1/2,color='gray',alpha=.08)
            for period in [8,6,5,4,3,2]:ax.axvline(1/period,color='gray',lw=.4,alpha=.5)
            ax.set(xlim=(.09,.55),xlabel='Frequency (cycles/year)',ylabel='PSD / integrated 2–8 year power',title=title)
            ax.set_xticks([1/8,1/6,1/5,1/4,1/3,1/2],[f'{1/t:.3g}\n({t}y)' for t in [8,6,5,4,3,2]])
            ax.legend(fontsize=8)
        fig.suptitle(f'2–8 year filtered first | July 1999–June 2021 | {method}\n22 years; native frequency spacing 0.0455 cycles/year; normalized shapes, not absolute strength')
        fig.savefig(FIG/f'S01_frequency_overview_{tag}.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,scope in zip(axs,['basin_500_3000','basin_1000_3000']):
        for i in [1,2,3]:
            f,p=spectra[(f'GL12_{scope}_CEOF{i}','Hann')]
            ax.plot(f,p/bandpower(f,p,1/8,1/2),'o-',ms=3,label=f'CEOF{i}')
        ax.set(xlim=(.09,.55),xlabel='Frequency (cycles/year)',ylabel='Normalized PSD',title=scope);ax.legend()
    fig.suptitle('CMEMS depth-limited CEOF: leading three time-coefficient spectra | Hann')
    fig.savefig(FIG/'S02_leading_modes.png',dpi=180);plt.close(fig)
    checks.update(period=['1999-07','2021-06'],samples=len(dates),frequency_spacing_per_year=12/len(dates),DPSS_NW=1.5,DPSS_K=2,DPSS_half_bandwidth_per_year=1.5/(len(dates)/12),all_inputs_bandpass_2_8_first=True,no_zero_padding=True,complex_TC_spectrum='one-sided P(real)+P(imag), not only real component',subband_power_definition='linear interpolation of native PSD to band limits, trapezoidal integral; normalized within2–8 years',no_formal_peak_significance_claim=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
