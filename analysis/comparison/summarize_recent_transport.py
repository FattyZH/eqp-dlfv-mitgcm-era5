"""Recent-window transport assessment using the existing long-record filter."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--start',default='2015-12-01');args=p.parse_args();base=ROOT/'research/transport/2026-11-13_1900m_ctrl';start=pd.Timestamp(args.start);dest=base/('since_'+start.strftime('%Y%m'));dest.mkdir(exist_ok=True);raw=pd.read_csv(base/'transports.csv',index_col=0,parse_dates=True);f=pd.read_csv(base/'filtered_transports.csv',index_col=0,parse_dates=True);assert raw.index.equals(f.index);rows=[]
 for trim in [0,24,36,48]:
  valid=f.iloc[trim:-trim] if trim else f;v=valid.loc[start:];m=v.model_common_net_Sv.values;r=v.reference_common_net_Sv.values;a=raw.loc[v.index];rows.append(dict(trim_months=trim,first_month=str(v.index[0].date()),last_month=str(v.index[-1].date()),months=len(v),correlation=float(np.corrcoef(m,r)[0,1]),model_std_Sv=float(np.std(m)),reference_std_Sv=float(np.std(r)),std_ratio=float(np.std(m)/np.std(r)),anomaly_rmse_Sv=float(np.sqrt(np.mean((m-r)**2))),model_mean_same_period_Sv=float(a.model_common_net_Sv.mean()),reference_mean_same_period_Sv=float(a.reference_common_net_Sv.mean()),endpoint_affected=trim==0))
 recent=raw.loc[start:];summary=dict(start=str(start.date()),raw_full_period=dict(first_month=str(recent.index[0].date()),last_month=str(recent.index[-1].date()),months=len(recent),model_mean_Sv=float(recent.model_common_net_Sv.mean()),reference_mean_Sv=float(recent.reference_common_net_Sv.mean()),raw_monthly_correlation=float(recent.model_common_net_Sv.corr(recent.reference_common_net_Sv))),filtered_windows=rows,method='Use previously filtered 1996-07 to2024-06 records; select recent period after endpoint exclusion; no short-record refiltering; identical common wet area.',caveat='Recent record has few independent 2-8 year cycles. Untrimmed filtered results include endpoint effects and are not main validation.')
 (dest/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
 fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
 for k,color,label in [('model_common_net_Sv','tab:orange','CTRL'),('reference_common_net_Sv','black','GLORYS')]:
  axs[0].plot(recent.index,recent[k],color=color,label=label);axs[1].plot(f.loc[start:].index,f.loc[start:,k],color=color,label=label)
 for ax in axs:ax.axhline(0,color='gray',lw=.7);ax.set(ylabel='Eastward transport (Sv)');ax.legend()
 edge=f.index[-37];axs[1].axvspan(edge,f.index[-1],color='gray',alpha=.2,label='Final 36 months: filter edge');axs[0].set_title('Monthly net transport, since '+start.strftime('%Y-%m'));axs[1].set_title(f'Long-record 2–8 yr filter | to {edge:%Y-%m}: r={rows[2]["correlation"]:.2f}, std ratio={rows[2]["std_ratio"]:.2f}');axs[1].legend();fig.savefig(dest/'recent_transport.png',dpi=180);plt.close(fig);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
