"""Separate spectral-record-length and short-record-filter effects on Q."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.ndimage import convolve1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_depth_amplitude import filt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/mechanism/2026-11-19_short_record_spectrum'
FIG=ROOT/'fig/261119_short_record_spectrum'

def iafilt(y):
    y=y.copy();ave=0
    for i in range(365):
        mean=y[i::365].mean();ave+=mean;y[i::365]-=mean
    return convolve1d(y+ave/365,np.ones(365)/365)

def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'research/mechanism/2026-11-17_historical_bandpass/transport_series.csv',parse_dates=['time']).set_index('time')
    raw=data['1900-bottom_GL12_common'];long=filt(raw.to_numpy()[:,None],raw.index.month.to_numpy())[:,0]
    recent=(raw.index>='2015-12-01')&(raw.index<='2024-06-01')
    short_raw=raw[recent]
    short=filt(short_raw.to_numpy()[:,None],short_raw.index.month.to_numpy())[:,0]
    daily_index=pd.date_range(short_raw.index[0],short_raw.index[-1]+pd.offsets.MonthEnd(0),freq='D')
    daily=short_raw.reindex(daily_index.to_period('M').to_timestamp()).to_numpy()
    short_ia=pd.Series(iafilt(daily),index=daily_index).resample('MS').mean().to_numpy()
    original=pd.read_csv(ROOT/'research/mechanism/2026-11-17_tke_floor/recent_iafilt_Sv.csv',parse_dates=['time']).set_index('time')
    signals={
        'CMEMS_common_long_filter_main22y':(long[36:-36],12),
        'CMEMS_common_long_filter_all28y_edge_sensitivity':(long,12),
        'CMEMS_common_long_filter_recent103mo':(long[recent],12),
        'CMEMS_common_short_filter_recent103mo':(short,12),
        'CMEMS_common_iafilt_recent103mo':(short_ia,12),
        'CMEMS_native_previous_iafilt_recent103mo':(original.GL12.to_numpy(),12),
        'OBS_previous_iafilt_recent103mo':(original.obs.to_numpy(),12),
    }
    with np.load(ROOT/'data/moor_transport.npz') as a:
        signals['OBS_original_daily_tp_smooth']=(a['tp_smooth'],365.25)
    rows=[];spectra={};curves=[]
    for key,(y,fs) in signals.items():
        for win in ['hann','boxcar']:
            f,p=periodogram(y,fs=fs,window=win,detrend='constant')
            mask=(f>=1/8)&(f<=1/2);peak=f[mask][np.argmax(p[mask])]
            df=fs/len(y)
            rows.append(dict(signal=key,window=win,samples=len(y),years_N_over_fs=len(y)/fs,frequency_spacing=df,peak_frequency_per_year=peak,peak_period_years=1/peak,second_bin_period_years=len(y)/fs/2))
            spectra[(key,win)]=(f,p)
            curves.extend(dict(signal=key,window=win,frequency=float(ff),PSD=float(pp)) for ff,pp in zip(f,p))
    pd.DataFrame(rows).to_csv(OUT/'peak_comparison.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'spectral_curves.csv',index=False)
    labels=[['CMEMS_common_long_filter_main22y','CMEMS_common_long_filter_recent103mo'],['CMEMS_common_long_filter_recent103mo','CMEMS_common_short_filter_recent103mo','CMEMS_common_iafilt_recent103mo'],['OBS_previous_iafilt_recent103mo','CMEMS_native_previous_iafilt_recent103mo'],['CMEMS_common_long_filter_recent103mo','CMEMS_common_short_filter_recent103mo']]
    titles=['Same CMEMS common-area Q: long vs recent window','Same recent CMEMS Q: processing sensitivity','Existing 103-month iafilt observation comparison','Recent Q time series: filter support sensitivity']
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for ax,keys,title in zip(axs.flat,labels,titles):
        if ax is axs[1,1]:
            ax.plot(short_raw.index,long[recent],label='Filter full28y then cut recent')
            ax.plot(short_raw.index,short,label='Filter recent103mo only')
            ax.set_ylabel('Bandpassed Q (Sv)')
        else:
            for key in keys:
                f,p=spectra[(key,'hann')];use=(f>=1/8)&(f<=1/2)
                ax.plot(f,p/p[use].max(),'o-',ms=4,label=key.replace('CMEMS_common_','').replace('_recent103mo','').replace('previous_iafilt','iafilt'))
            ax.set(xlim=(.09,.55),xlabel='Frequency (cycles/year)',ylabel='PSD / maximum within2–8 years')
            ax.axvline(1/(103/12/2),color='gray',ls=':',label='103-month second bin:4.29y')
            ax.set_xticks([1/8,1/6,1/4.5,1/3,1/2],['0.125\n8y','0.167\n6y','0.222\n4.5y','0.333\n3y','0.5\n2y'])
        ax.set_title(title);ax.legend(fontsize=7)
    fig.suptitle('Period estimates depend on record length, period and smoothing | Hann, native FFT bins\nRecent window: December2015–June2024; full-support recent tail includes filter-edge influence')
    fig.savefig(FIG/'W01_window_filter_comparison.png',dpi=180);plt.close(fig)
    (OUT/'definitions.json').write_text(json.dumps(dict(primary_long_period='1999-07 to2021-06',recent_period='2015-12 to2024-06,103months',long_filter_support='1996-07 to2024-06',recent_tail_has_unavoidable_end_of_source_filter_edges=True,short_filter_all103months_shown_for_sensitivity_not_new_main_analysis=True,iafilt_common='Short monthly Q expanded to days from2015-12-01 to2024-06-30; original365sample climate removal and mean restore;365day reflect convolution; then monthly mean',previous_iafilt='Previously validated three-way observation comparison; native geometry differs from common-area Q',do_not_interpret_second_bin_peak_as_proof_of_periodicity=True),indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
