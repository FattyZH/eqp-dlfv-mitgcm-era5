"""Mooring/CTRL/GLORYS transport: identical user-defined 365-day smoother.

Monthly model/reanalysis means contain no daily variability. The main comparison
therefore aggregates the mooring to complete months, expands EVERY source as
piecewise-constant monthly means on one daily calendar, and applies the same
iafilt. The supplied daily mooring smoother is reproduced independently.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
ALGORITHM='moor_threeway_user_iafilt_v1'
SOURCES={'obs':'Mooring','ctrl_native':'CTRL','glorys_available':'GLORYS','ctrl_common':'CTRL common area','glorys_common':'GLORYS common area'}
COLORS={'obs':'black','ctrl_native':'#D55E00','glorys_available':'#0072B2'}


def iafilt(data,axis=0,fs=365,rmclim=True):
    """User's algorithm, including mean restoration and scipy reflect boundary."""
    data=np.asarray(data,dtype=float).copy();data=np.moveaxis(data,axis,0)
    if data.shape[0]<fs:raise ValueError('At least one full seasonal cycle is required')
    if not np.isfinite(data).all():raise ValueError('Do not interpolate missing values silently')
    if rmclim:
        ave=0
        for i in range(fs):
            ave+=data[i::fs].mean(0)
            data[i::fs]-=data[i::fs].mean(0)
        ave/=fs;data+=ave
    data=convolve1d(data,np.ones(fs)/fs,axis=0,mode='reflect',origin=0)
    return np.moveaxis(data,0,axis)


def stats(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Paired finite values required')
    return dict(n=len(x),mean_candidate_Sv=float(x.mean()),mean_reference_Sv=float(y.mean()),bias_Sv=float((x-y).mean()),std_candidate_Sv=float(x.std()),std_reference_Sv=float(y.std()),std_ratio=float(x.std()/y.std()),correlation=float(np.corrcoef(x,y)[0,1]),rmse_Sv=float(np.sqrt(np.mean((x-y)**2))),centered_rmse_Sv=float(np.sqrt(np.mean(((x-x.mean())-(y-y.mean()))**2))))


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mooring',type=Path,default=ROOT/'data/moor_transport.npz');p.add_argument('--transport',type=Path,default=ROOT/'research/transport/2026-11-13_1900m_ctrl/transports.csv');p.add_argument('--plot-output',type=Path,default=None);p.add_argument('--output',type=Path,default=ROOT/'research/transport/2026-11-13_mooring_threeway');args=p.parse_args();dest=args.output;dest.mkdir(parents=True,exist_ok=True);plot=args.plot_output or ROOT/'fig/261113_mooring_threeway';plot.mkdir(parents=True,exist_ok=True)
    with np.load(args.mooring,allow_pickle=False) as d:
        t=pd.DatetimeIndex(d['time']);tp=np.asarray(d['tp'],float).squeeze();supplied=np.asarray(d['tp_smooth'],float).squeeze()
    if tp.shape!=(len(t),) or supplied.shape!=tp.shape:raise ValueError('One-dimensional co-timed observation series required')
    if not t.is_unique or not t.is_monotonic_increasing:raise ValueError('Observation timestamps must be sorted and unique')
    if not t.equals(pd.date_range(t[0],t[-1],freq='D')):raise ValueError('Daily regular observations required; inspect gaps before analysis')
    if not np.isfinite(tp).all() or not np.isfinite(supplied).all():raise ValueError('Observation contains missing data')
    obs=pd.Series(tp,index=t,name='obs');saved=pd.Series(supplied,index=t,name='obs_supplied_smooth')
    reconstructed=iafilt(tp);reproduction_error=float(np.max(abs(reconstructed-supplied)))
    if reproduction_error>1e-8:raise ValueError(f'Supplied smoother differs from user function: max error {reproduction_error:g} Sv')
    observed_monthly=obs.resample('MS').mean();counts=obs.resample('MS').count();complete=counts==counts.index.days_in_month;observed_monthly=observed_monthly[complete]
    q=pd.read_csv(args.transport,index_col=0,parse_dates=True)
    monthly=pd.DataFrame(dict(obs=observed_monthly)).join(q[['model_native_net_Sv','reference_available_net_Sv','model_common_net_Sv','reference_common_net_Sv']].rename(columns={'model_native_net_Sv':'ctrl_native','reference_available_net_Sv':'glorys_available','model_common_net_Sv':'ctrl_common','reference_common_net_Sv':'glorys_common'}),how='inner')
    if monthly.isna().any().any():raise ValueError('Missing input month; cannot silently bridge gaps')
    if not monthly.index.equals(pd.date_range(monthly.index[0],monthly.index[-1],freq='MS')):raise ValueError('Common complete months have a gap')
    end=monthly.index[-1]+pd.offsets.MonthBegin(1);days=pd.date_range(monthly.index[0],end-pd.Timedelta(days=1),freq='D');daily=monthly.reindex(days,method='ffill');smooth=pd.DataFrame(iafilt(daily.values),index=days,columns=monthly.columns)
    full=pd.Series(False,index=days);full.iloc[182:-182]=True;counts_valid=full.resample('MS').sum();valid_months=counts_valid.index[counts_valid.values==counts_valid.index.days_in_month];sm_month=smooth.resample('MS').mean();valid=sm_month.loc[valid_months]
    # The input mean is restored by iafilt; smoothed Q is NOT a zero-mean bandpass.
    monthly.to_csv(dest/'monthly_raw_Sv.csv',index_label='time');out=smooth.copy();out['full_365day_window']=full;out.to_csv(dest/'daily_smoothed_Sv.csv',index_label='time');out=sm_month.copy();out['full_window_all_days']=out.index.isin(valid_months);out['obs_supplied_smooth']=saved.resample('MS').mean().reindex(out.index);out.to_csv(dest/'monthly_smoothed_Sv.csv',index_label='time')
    rows=[]
    for label,frame in [('raw_monthly',monthly),('smoothed365_full_period',sm_month),('smoothed365_complete_windows',valid)]:
        for candidate,reference in [('ctrl_native','obs'),('glorys_available','obs'),('ctrl_native','glorys_available'),('ctrl_common','obs'),('glorys_common','obs'),('ctrl_common','glorys_common')]:
            rows.append(dict(processing=label,candidate=candidate,reference=reference,first_month=str(frame.index[0].date()),last_month=str(frame.index[-1].date()),**stats(frame[candidate],frame[reference])))
    metrics=pd.DataFrame(rows);metrics.to_csv(dest/'metrics.csv',index=False)
    # Matched temporal sampling is the primary comparison. Quantify the loss from
    # monthly aggregation against the ORIGINAL daily mooring smoother separately.
    supplied_month=saved.resample('MS').mean().reindex(valid.index);sampling=stats(valid.obs,supplied_month);sampling['maximum_absolute_difference_Sv']=float(np.max(abs(valid.obs-supplied_month)))
    # Full-window annual averaging removes a fixed 365-sample cycle exactly, so
    # the user's mean-restored filter equals raw365 box averaging in the interior.
    raw_box=convolve1d(tp,np.ones(365)/365,mode='reflect');interior_identity_error=float(np.max(abs(reconstructed[182:-182]-raw_box[182:-182])))
    if interior_identity_error>1e-10:raise AssertionError('Interior annual-box identity')
    if not np.allclose(iafilt(np.full((800,2),3.25)),3.25):raise AssertionError('Constant-input/mean restoration')
    test=np.stack([tp,tp*2],axis=1);axis_error=float(np.max(abs(iafilt(test.T,axis=1).T-iafilt(test))))
    if axis_error>1e-12:raise AssertionError('Axis consistency')
    # Paired circular blocks are conditional time-sampling ranges, not an
    # independent mooring/reanalysis error model or an initial-condition ensemble.
    rng=np.random.default_rng(20261113);N=len(valid);bootstrap=[]
    for block in [12,24]:
        samples=[]
        for _ in range(2000):
            ix=((rng.integers(N,size=(int(np.ceil(N/block)),1))+np.arange(block))%N).ravel()[:N];a=valid.iloc[ix];cs=stats(a.ctrl_native,a.obs);gs=stats(a.glorys_available,a.obs);samples.append([cs['correlation'],gs['correlation'],cs['std_ratio'],gs['std_ratio'],cs['correlation']-gs['correlation']])
        lo,hi=np.quantile(samples,[.025,.975],axis=0)
        for j,name in enumerate(['ctrl_obs_correlation','glorys_obs_correlation','ctrl_obs_std_ratio','glorys_obs_std_ratio','correlation_difference_ctrl_minus_glorys']):bootstrap.append(dict(block_months=block,metric=name,ci_low=lo[j],ci_high=hi[j]))
    pd.DataFrame(bootstrap).to_csv(dest/'paired_time_uncertainty.csv',index=False)
    validation=dict(supplied_iafilt_max_error_Sv=reproduction_error,original_daily_sample_count=len(t),regular_daily_no_missing=True,common_complete_months=len(monthly),common_daily_samples=len(days),complete_window_months=len(valid),complete_window_daily_samples=int(full.sum()),fixed365_raw_box_interior_identity_error_Sv=interior_identity_error,axis_consistency_max_error_Sv=axis_error,constant_input_preserved=True,all_sources_same_operator=True,monthly_to_daily='Piecewise constant monthly means for ALL sources; no invented submonthly variability',units_and_sign='Sv/eastward positive inferred from user-provided same-geometry transport convention; NPZ has no embedded unit metadata')
    (dest/'validation.json').write_text(json.dumps(validation,indent=2))
    info=dict(algorithm=ALGORITHM,definition='142E, New Guinea coast to2N, below1900m; eastward positive',observation=dict(file=str(args.mooring.relative_to(ROOT)),first_day=str(t[0].date()),last_day=str(t[-1].date()),samples=len(t),raw_mean_Sv=float(tp.mean()),provided_smooth_mean_Sv=float(supplied.mean()),provided_smooth_std_Sv=float(supplied.std())),common_months=dict(first=str(monthly.index[0].date()),last=str(monthly.index[-1].date()),count=len(monthly)),main_smoothing_months=dict(first=str(sm_month.index[0].date()),last=str(sm_month.index[-1].date()),count=len(sm_month)),means_raw_monthly_Sv={k:float(v) for k,v in monthly.mean().items()},means_raw_daily_timeweighted_Sv={k:float(v) for k,v in daily.mean().items()},main_metrics=metrics.query("processing=='smoothed365_full_period'").to_dict('records'),raw_metrics=metrics.query("processing=='raw_monthly'").to_dict('records'),mooring_monthly_sampling_effect=sampling,filter=dict(fs=365,rmclim=True,seasonal_definition='sample index modulo365 from the common daily start; not calendar month/day climatology',mean_restored=True,window='365 samples, centered',boundary_mode='scipy.ndimage.convolve1d default reflect/origin0',statistics_exclude_edges_days=0,complete_window_sensitivity_trim_days=182,output='mean-retaining 365-day smoothed transport, NOT 2-8 year bandpass'),geometry_caveat='Observation NPZ provides integrated Q only and cannot be cropped to model/GLORYS common wet mask. Primary observation comparisons use model native full-bottom and GLORYS available native areas; prior common-area series are sensitivity checks, not a matched observation area.',uncertainty='Paired circular time blocks, not independent samples or observational error estimates; observation spatial reconstruction/independence remains to be checked.')
    (dest/'summary.json').write_text(json.dumps(info,indent=2))
    # The supplied daily record is shown lightly; solid black is the matched
    # monthly-sampling result used in main statistics.
    fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for k in ['obs','ctrl_native','glorys_available']:
        axs[0].plot(monthly.index,monthly[k],color=COLORS[k],label=SOURCES[k],lw=1.2);axs[1].plot(smooth.index,smooth[k],color=COLORS[k],label=SOURCES[k],lw=1.4)
    for ax in axs:ax.axhline(0,color='gray',lw=.6);ax.set(ylabel='Eastward transport (Sv)');ax.legend(ncol=3)
    axs[0].set_title('Raw monthly means: complete shared months');axs[1].set_title('User iafilt: mean restored + centered365 days; gray: reflected edge only')
    axs[1].axvspan(days[0],days[182],color='gray',alpha=.18);axs[1].axvspan(days[-183],days[-1],color='gray',alpha=.18);fig.suptitle('Mooring / CTRL / GLORYS |142 E, below1900m');fig.savefig(plot/'M01_threeway_transport.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained',sharex=True,sharey=True)
    for ax,k in zip(axs,['ctrl_native','glorys_available']):
        st=stats(valid[k],valid.obs);ax.scatter(valid.obs,valid[k],s=15,alpha=.7,color=COLORS[k]);lims=[min(valid.obs.min(),valid[k].min()),max(valid.obs.max(),valid[k].max())];ax.plot(lims,lims,'k--',lw=.7);ax.set(title=f'{SOURCES[k]}: r={st["correlation"]:.2f}, std/obs={st["std_ratio"]:.2f}',xlabel='Mooring smoothed transport (Sv)',ylabel='Candidate smoothed transport (Sv)')
    fig.suptitle(f'Complete-window monthly values |{valid.index[0]:%Y-%m} to{valid.index[-1]:%Y-%m}');fig.savefig(plot/'M02_smoothed_scatter.png',dpi=180);plt.close(fig)
    manifest=dict(algorithm=ALGORITHM,input_sha256={str(q.relative_to(ROOT)):sha(q) for q in [args.mooring,args.transport]},script_sha256=sha(Path(__file__)),artifacts=[str(q.relative_to(ROOT)) for q in sorted(dest.iterdir()) if q.is_file()])
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(info,indent=2),flush=True);print('VALIDATION',json.dumps(validation),flush=True)


if __name__=='__main__':main()
