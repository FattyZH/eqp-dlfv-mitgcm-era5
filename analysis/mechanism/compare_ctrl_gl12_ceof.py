"""Independent CTRL/GL12 CEOFs with identical grid, mask and RF weighting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compute_ctrl_ceof import decompose
from diagnose_depth_amplitude import filt

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/mechanism/2026-11-16_depth_amplitude'
OUT=ROOT/'research/mechanism/2026-11-17_ctrl_gl12_ceof'
FIG=ROOT/'fig/261117_ctrl_gl12_ceof'

def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'source_sections.npz') as a:
        arrays={'GL12':a['gl12'],'CTRL':a['model']}
        time=pd.DatetimeIndex(a['time']);kk=a['kk']
    with xr.open_dataset(SOURCE/'section_statistics.nc') as d:
        x=d.longitude.values;z=d.depth.values
    mask=np.logical_and.reduce([np.isfinite(v).all(axis=0) for v in arrays.values()])
    filtered={k:filt(v,time.month.to_numpy()) for k,v in arrays.items()}
    mask&=np.logical_and.reduce([v[36:-36].std(axis=0)>1e-12 for v in filtered.values()])
    rf=-np.fromfile(ROOT/'output/261016_ctrl1/RF.data',dtype='>f4').astype(float)
    dz=np.maximum(0,np.minimum(rf[kk+1],3000)-rf[kk])
    weights=np.broadcast_to(dz[:,None],mask.shape)
    time=time[36:-36];result={};checks={};rows=[]
    for key,v in filtered.items():
        a=np.full(v.shape,np.nan+1j*np.nan)
        a[:,mask]=hilbert(v[:,mask],axis=0)
        sm,tc,fractions,check=decompose(a[36:-36],mask,weights)
        result[key]=dict(sm=sm,tc=tc,fractions=fractions)
        checks[key]=check
        for i in range(4):rows.append(dict(dataset=key,mode=i+1,explained_variance=float(fractions[i]),cumulative_variance=float(fractions[:i+1].sum())))
    w=weights[mask];gl=result['GL12'];ctrl=result['CTRL']
    # Spatial-overlap phase alignment only: no lag fitting or optimizing r.
    rotations=[]
    for i in range(4):
        overlap=np.sum(np.conj(gl['sm'][i,mask])*ctrl['sm'][i,mask]*w)
        rot=np.exp(-1j*np.angle(overlap))
        before=ctrl['tc'][:,i,None]*ctrl['sm'][i,mask][None]
        ctrl['sm'][i]*=rot;ctrl['tc'][:,i]*=np.conj(rot)
        err=float(np.max(abs(before-ctrl['tc'][:,i,None]*ctrl['sm'][i,mask][None])))
        assert err<1e-12
        rotations.append(dict(mode=i+1,spatial_rotation_degrees=float(np.angle(rot,deg=True)),reconstruction_change_max_m_s=err))
    pairs=[]
    for i in range(4):
        for j in range(4):
            a=gl['sm'][i,mask];b=ctrl['sm'][j,mask]
            overlap=np.sum(np.conj(a)*b*w)/np.sqrt(np.sum(abs(a)**2*w)*np.sum(abs(b)**2*w))
            t=gl['tc'][:,i];u=ctrl['tc'][:,j]
            coherence=np.vdot(t,u)/np.sqrt(np.vdot(t,t).real*np.vdot(u,u).real)
            pairs.append(dict(GL12_mode=i+1,CTRL_mode=j+1,spatial_complex_congruence_abs=float(abs(overlap)),temporal_complex_correlation_abs=float(abs(coherence)),temporal_complex_phase_deg=float(np.angle(coherence,deg=True)),aligned_real_TC_r=float(np.corrcoef(t.real,u.real)[0,1])))
    for key,r in result.items():
        np.savez_compressed(OUT/f'{key}_ceof_RF_common.npz',spatial_mode=r['sm'],time_coefficient=r['tc'],explained_variance=r['fractions'],mask=mask,weights=weights,longitude=x,depth=z,time=time.values)
        xr.Dataset(dict(spatial_real=(('mode','depth','longitude'),r['sm'].real),spatial_imag=(('mode','depth','longitude'),r['sm'].imag),time_real=(('time','mode'),r['tc'].real),time_imag=(('time','mode'),r['tc'].imag),explained_variance=(('mode',),r['fractions'][:4]),common_mask=(('depth','longitude'),mask),RF_weight_m=(('depth','longitude'),weights)),coords=dict(mode=np.arange(1,5),time=time,depth=z,longitude=x),attrs=dict(dataset=key,spatial_units='m/s',TC_units='dimensionless',reconstruction='Real(TC @ spatial_mode)',phase='lag phase = -arg(spatial_mode); GL12 TC[0] real positive; CTRL aligned by RF-weighted spatial overlap of same-index modes',method='Independent CEOFs; same mask and RF thickness weights; sqrt(weights) in covariance; no pointwise standardization',period='1999-07 to 2021-06',filter='Own monthly climatology removed; fourth-order zero-phase 2–8 year; Hilbert full support 1996-07 to 2024-06 before 36-month trim; center complex anomalies after trim')).to_netcdf(OUT/f'{key}_ceof_RF_common.nc')
    pd.DataFrame(rows).to_csv(OUT/'explained_variance.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'mode_pair_comparison.csv',index=False)
    for i in [0,1]:
        for tag,top in [('full',0),('deep',1000)]:
            fig,axs=plt.subplots(3,2,figsize=(14,11),layout='constrained',height_ratios=[2,2,1])
            visible=z>=top
            amps=np.array([abs(result[k]['sm'][i])*100 for k in ['GL12','CTRL']])
            vmax=float(np.nanpercentile(amps[:,visible],99))
            threshold=.05*np.nanmax(amps)
            for col,key in enumerate(['GL12','CTRL']):
                r=result[key];amp=abs(r['sm'][i])*100
                phase=np.where(amp>=threshold,-np.angle(r['sm'][i],deg=True),np.nan)
                im=axs[0,col].pcolormesh(x,z,amp,cmap='viridis',vmin=0,vmax=vmax,shading='nearest')
                axs[0,col].set_title(f'{key} CEOF{i+1} | {r["fractions"][i]*100:.2f}%')
                pm=axs[1,col].pcolormesh(x,z,phase,cmap='twilight_shifted',vmin=-180,vmax=180,shading='nearest')
                axs[1,col].set_title(f'{key} spatial lag phase')
                for row in [0,1]:
                    ax=axs[row,col];ax.set(xlim=(130,285),ylim=(3000,top),xlabel='Longitude (°E)',ylabel='Depth (m)');ax.set_facecolor('.85')
                    if top==0:ax.axhline(300,color='magenta',ls='--',lw=.7)
                    ax.axhline(1900,color='gray',ls=':',lw=.7)
            fig.colorbar(im,ax=list(axs[0]),label='Complex spatial-mode magnitude (cm/s)',extend='max',shrink=.85)
            fig.colorbar(pm,ax=list(axs[1]),label='−arg(mode), degrees; shared 5% magnitude mask',shrink=.85)
            for key in ['GL12','CTRL']:
                axs[2,0].plot(time,result[key]['tc'][:,i].real,label=key)
                axs[2,1].plot(time,result[key]['tc'][:,i].imag,label=key)
            for col,title in enumerate(['Real TC','Imaginary TC']):
                axs[2,col].set(title=title,ylabel='Dimensionless coefficient');axs[2,col].axhline(0,color='gray',lw=.7);axs[2,col].legend()
            fig.suptitle(f'Independent CEOF{i+1} | identical RF weights and wet mask | equatorial ±0.5°\n2–8 years | July 1999–June 2021 | CTRL phase aligned by spatial overlap only',fontsize=13)
            fig.savefig(FIG/f'G0{i+1}_ceof_{tag}.png',dpi=180);plt.close(fig)
    checks.update(common_points=int(mask.sum()),time_samples=len(time),same_weights_and_mask=True,phase_alignment=rotations,source_sha256=hashlib.sha256((SOURCE/'source_sections.npz').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(pairs).query('GL12_mode==CTRL_mode').to_string(index=False),flush=True)

if __name__=='__main__':main()
