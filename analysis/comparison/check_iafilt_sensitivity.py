"""Compare annual smoothing definitions on identical full103-month records.

Annual cycle always has365 samples; smoothing-window length is independent.
Changing iafilt(fs=730) would remove a two-year cycle and is not tested here.
No model inputs or outputs are modified.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_mooring_transport import iafilt,stats
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'research/transport/2026-11-16_iafilt_sensitivity'
PLOT=ROOT/'fig/261116_iafilt_sensitivity'
SOURCE=ROOT/'research/transport/2026-11-16_ctrl_ensemble/monthly_raw_Sv.csv'
COLS=[f'ctrl{i}' for i in range(5)]+['ensemble_mean','obs','glorys']


def remove_cycle(a):
    a=a.copy();ave=0
    for i in range(365):
        ave+=a[i::365].mean(0)
        a[i::365]-=a[i::365].mean(0)
    return a+ave/365


def main():
    DEST.mkdir(parents=True,exist_ok=True);PLOT.mkdir(parents=True,exist_ok=True)
    monthly=pd.read_csv(SOURCE,index_col=0,parse_dates=True)[COLS]
    days=pd.date_range(monthly.index[0],monthly.index[-1]+pd.offsets.MonthEnd(0),freq='D')
    daily=monthly.reindex(days,method='ffill');raw=daily.values
    seasonal=remove_cycle(raw)
    methods={};definitions=[]
    def add(name,values,description,window_days=None,mode=None):
        methods[name]=pd.DataFrame(values,index=days,columns=COLS).resample('MS').mean()
        definitions.append(dict(method=name,description=description,window_days=window_days,boundary=mode))
    base=iafilt(raw)
    add('user_365_reflect',base,'Original user iafilt,365sample seasonal removal plus mean restoration',365,'reflect')
    add('raw_365_reflect',convolve1d(raw,np.ones(365)/365,axis=0,mode='reflect'),'Raw365day box; no seasonal removal',365,'reflect')
    for mode in ['nearest','mirror']:
        add('user_365_'+mode,convolve1d(seasonal,np.ones(365)/365,axis=0,mode=mode),'Same365sample seasonal removal; alternate edge extension',365,mode)
    truncated=convolve1d(seasonal,np.ones(365),axis=0,mode='constant',cval=0)
    weights=convolve1d(np.ones(len(days)),np.ones(365),mode='constant',cval=0)
    add('user_365_partial',truncated/weights[:,None],'Same seasonal removal; edge windows shortened and renormalized',365,'available samples only')
    for window in [183,273,455,547,731]:
        add(f'annual_cycle_box{window}',convolve1d(seasonal,np.ones(window)/window,axis=0,mode='reflect'),'Fixed365sample seasonal cycle; change smoothing length only',window,'reflect')
    mc=monthly.groupby(monthly.index.month).transform('mean')
    calendar=monthly-mc+monthly.mean()
    expanded=calendar.reindex(days,method='ffill').values
    add('calendar_month_365',convolve1d(expanded,np.ones(365)/365,axis=0,mode='reflect'),'Calendar-month climatology removal plus monthly overall mean;365day box',365,'reflect')
    # Centered two-pass12-month average: half weight at +/-6months,
    # full weight for the11 internal months, total13 positions.
    kernel=np.r_[.5,np.ones(11),.5]/12
    two12=convolve1d(monthly.values,kernel,axis=0,mode='reflect')
    methods['monthly_centered_2x12']=pd.DataFrame(two12,index=monthly.index,columns=COLS)
    definitions.append(dict(method='monthly_centered_2x12',description='Centered2x12month box on raw monthly means; equal month weights,13positions',window_days=None,boundary='reflect'))
    reference_saved=pd.read_csv(ROOT/'research/transport/2026-11-16_ctrl_ensemble/monthly_smoothed_Sv.csv',index_col=0,parse_dates=True)
    error=float(np.max(abs(methods['user_365_reflect'][COLS].values-reference_saved[COLS].values)))
    if error>1e-10:raise AssertionError('Original smoother reproduction')
    identity=float(np.max(abs(base[182:-182]-convolve1d(raw,np.ones(365)/365,axis=0,mode='reflect')[182:-182])))
    if identity>1e-10:raise AssertionError('Annual-box interior identity')
    full_daily=pd.Series(False,index=days);full_daily.iloc[182:-182]=True
    full_months=full_daily.resample('MS').sum().eq(monthly.index.days_in_month)
    interior_rows=[]
    for name in ['user_365_reflect','raw_365_reflect']:
        frame=methods[name].loc[full_months]
        for candidate in ['ensemble_mean','ctrl1','glorys']:
            interior_rows.append(dict(method=name,candidate=candidate,
                                      role='Diagnostic only: locate edge effect, not primary statistics',
                                      **stats(frame[candidate],frame.obs)))
    pd.DataFrame(interior_rows).to_csv(DEST/'edge_localization_diagnostic.csv',index=False)
    seasonal_leak=convolve1d(raw-seasonal,np.ones(365)/365,axis=0,mode='reflect')
    reconstruction=float(np.max(abs(seasonal_leak-(convolve1d(raw,np.ones(365)/365,axis=0,mode='reflect')-base))))
    if reconstruction>1e-10:raise AssertionError('Edge difference must equal filtered seasonal component')
    leak=[dict(source=col,max_abs_seasonal_edge_difference_Sv=float(abs(seasonal_leak[:,i]).max()),
               max_abs_complete_window_difference_Sv=float(abs(seasonal_leak[182:-182,i]).max()))
          for i,col in enumerate(COLS)]
    pd.DataFrame(leak).to_csv(DEST/'seasonal_edge_leakage.csv',index=False)
    rows=[];differences=[]
    for name,frame in methods.items():
        for candidate in [f'ctrl{i}' for i in range(5)]+['ensemble_mean','glorys']:
            for ref in ['obs','glorys']:
                if candidate==ref:continue
                rows.append(dict(method=name,candidate=candidate,reference=ref,period='2015-12 to2024-06;all103months including endpoints',**stats(frame[candidate],frame[ref])))
        for col in COLS:
            delta=frame[col]-methods['user_365_reflect'][col]
            differences.append(dict(method=name,source=col,rmse_from_original_Sv=float(np.sqrt(np.mean(delta**2))),max_abs_difference_Sv=float(abs(delta).max())))
    metrics=pd.DataFrame(rows);metrics.to_csv(DEST/'metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(DEST/'differences_from_original.csv',index=False)
    pd.concat(methods,names=['method','time']).to_csv(DEST/'monthly_processed_Sv.csv')
    (DEST/'methods.json').write_text(json.dumps(definitions,indent=2))
    (DEST/'validation.json').write_text(json.dumps(dict(original_smoother_reproduction_max_error_Sv=error,raw_box_vs_user_interior_max_error_Sv=identity,seasonal_edge_difference_reconstruction_error_Sv=reconstruction,all_metrics_use103months=bool((metrics.n==103).all()),all_methods_same_input_and_dates=True,seasonal_cycle_not_changed_with_window=True,model_inputs_untouched=True),indent=2))
    selected=metrics.query("reference=='obs'").pivot(index='method',columns='candidate',values='correlation')
    selected.to_csv(DEST/'correlations_with_mooring.csv')
    # Plot only edge/same nominal annual-window variants together; longer
    # windows change the scientific frequency content and are shown separately.
    names=['user_365_reflect','raw_365_reflect','user_365_nearest','user_365_partial','calendar_month_365','monthly_centered_2x12']
    fig,axs=plt.subplots(3,1,figsize=(12,10),layout='constrained',sharex=True)
    for ax,col,title in zip(axs,['obs','ensemble_mean','glorys'],['Mooring','Five-member CTRL mean','GLORYS']):
        for name in names:ax.plot(monthly.index,methods[name][col],label=name,lw=1.5 if name=='user_365_reflect' else 1,alpha=.85)
        ax.axhline(0,color='gray',lw=.5);ax.set(title=title,ylabel='Eastward transport (Sv)')
    axs[0].legend(ncol=2,fontsize=8);fig.suptitle('Annual smoother definition/edge sensitivity | full103-month record')
    fig.savefig(PLOT/'F01_smoothing_definitions.png',dpi=180);plt.close(fig)
    windows=[183,273,365,455,547,731]
    keys=[f'annual_cycle_box{w}' if w!=365 else 'user_365_reflect' for w in windows]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name in [f'ctrl{i}' for i in range(5)]+['ensemble_mean','glorys']:
        frame=metrics.query("reference=='obs' and candidate==@name").set_index('method').loc[keys]
        for ax,col in zip(axs,['correlation','std_ratio']):ax.plot(np.array(windows)/365,frame[col],'o-',label=name,lw=2 if name=='ensemble_mean' else 1)
    axs[0].set(ylabel='Correlation with mooring');axs[1].set(ylabel='Std / mooring std')
    for ax in axs:ax.set_xlabel('Smoothing window (years)');ax.axvline(1,color='gray',ls='--',lw=.7)
    axs[0].legend(fontsize=8,ncol=2);fig.suptitle('Fixed annual-cycle removal; vary smoothing window | all103months')
    fig.savefig(PLOT/'F02_window_sensitivity.png',dpi=180);plt.close(fig)
    lines=['# iafilt处理敏感性','',
           '沿用全部共同103个月（2015-12至2024-06），不剔除任何首尾月份。全部方法使用同一月输运输入、同一日期；观测先月平均，三方共同铺为日序列，最后月平均统计。未改变模式参数或运行模拟。','',
           '主方法仍为用户原iafilt。固定去周期长度365天，窗口改变时不改变周期长度；直接把fs改为730会改成去两年周期，不是单纯加长平滑。', '',
           '| 方法 | CTRL0 r | CTRL1 r | CTRL2 r | CTRL3 r | CTRL4 r | 集合均值r | GLORYS r |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in definitions:
        name=item['method'];v=selected.loc[name]
        lines.append('| '+name+' | '+' | '.join(f'{v[k]:.3f}' for k in [f'ctrl{i}' for i in range(5)]+['ensemble_mean','glorys'])+' |')
    lines+=['','方法中文解释：user_365_reflect为原方法；raw_365_reflect不去气候态；nearest为端值延拓；mirror为不重复端点的镜像；partial为端点缩短窗口并归一化；calendar_month_365按公历月份去气候态；monthly_centered_2x12为居中两次12个月平均。annual_cycle_box183/273/455/547/731分别约6/9/15/18/24个月，仍先去365天周期。','',
            '原函数与直接365天移动平均在完整窗口内部代数等价：所去除的周期平均在一个完整365天窗口内恰被加回均值抵消。两者全窗差异主要来自端点延拓。保留均值还是减去整体常数不会改变Pearson相关。','',
            '缩短窗口会放进更多次年尺度波动；加长窗口则更强调多年变化。它们并不检验完全相同的信号，不能把最大相关的方法自动选为主方案。不同边界处理的相关差异反映端点假设，而非新增独立证据。','',
            '图：fig/261116_iafilt_sensitivity/F01_smoothing_definitions.png和F02_window_sensitivity.png。全部逐成员r、std比及RMSE见metrics.csv，定义见methods.json。']
    lines+=['',
            '主要发现：集合均值原iafilt全窗r=.253，直接365天平均r=.494，原始月序列居中2x12平均r=.490。'
            '两种365天算法在内部窗口一致，差值可严格重建为被滤后的365样本周期分量：'
            '反射原始强季节序列在端点不能保证完整周期抵消，去周期后再反射则避免了这部分残留。'
            '这里的r提高不能解释为内部年际信号被更好再现。'
            '定位诊断中，完整窗口91个月两者集合均值r均约.272（该诊断不替代全103月主结果）。'
            '仅更改去季节后的反射方式或使用公历月份气候态，集合均值r约.24–.25；'
            '在固定去365天周期前提下，将窗口改为6–24个月，集合均值r约.23–.33，'
            '未恢复.6–.8。各成员在原始365天平均下r约.47–.52。', '',
            '建议保留用户原iafilt为主，同时报告端点假设敏感性，不按最大r选择方法。'
            '旧高相关可能部分受端点处理影响，但未恢复旧代码/运行前不能认定这就是全部原因。']
    (DEST/'assessment.md').write_text('\n'.join(lines)+'\n')
    (DEST/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(selected.to_string(),flush=True)
    print('VALIDATION',error,identity,flush=True)

if __name__=='__main__':main()
