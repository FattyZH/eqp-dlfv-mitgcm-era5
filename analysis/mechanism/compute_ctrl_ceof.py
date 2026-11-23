"""Recompute CTRL equatorial CEOF; retain equal-point convention and RF check."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_depth_amplitude import filt

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/mechanism/2026-11-16_depth_amplitude'
OUT=ROOT/'research/mechanism/2026-11-17_ctrl_ceof'
FIG=ROOT/'fig/261117_ctrl_ceof'

def decompose(analytic,mask,weights):
    n=analytic.shape[0]
    a=analytic[:,mask].copy()
    a-=a.mean(axis=0)
    w=weights[mask]
    w=w/w.mean()
    aw=a*np.sqrt(w)[None]
    values,u=np.linalg.eigh(aw@aw.conj().T)
    order=np.argsort(values)[::-1]
    values=np.maximum(values[order].real,0)
    u=u[:,order]
    tc=u[:,:4]*np.sqrt(n)
    sm=tc.conj().T@a/n
    rotation=np.exp(-1j*np.angle(tc[0]))
    tc*=rotation[None];sm*=np.conj(rotation)[:,None]
    out=np.full((4,*mask.shape),np.nan+1j*np.nan)
    out[:,mask]=sm
    reconstruction=tc@sm
    residual=a-reconstruction
    captured=1-np.sum(abs(residual)**2*w[None])/np.sum(abs(a)**2*w[None])
    closure=abs(captured-values[:4].sum()/values.sum())
    orthogonal=float(np.max(abs(tc.conj().T@tc/n-np.eye(4))))
    assert closure<1e-10 and orthogonal<1e-10
    # Full temporal eigenbasis reconstructs the original analytic anomalies.
    full=u@(u.conj().T@a)
    full_error=float(np.max(abs(full-a)))
    assert full_error<1e-10
    return out,tc,values/values.sum(),dict(four_mode_variance_closure_error=float(closure),temporal_orthogonality_error=orthogonal,full_basis_reconstruction_max_error_m_s=full_error)

def plot_modes(sm,tc,fractions,x,z,time,tag):
    for mode in [0,1]:
        amp=abs(sm[mode])*100
        phase=-np.angle(sm[mode],deg=True)
        phase=np.where(amp>=.05*np.nanmax(amp),phase,np.nan)
        fig=plt.figure(figsize=(13,8),layout='constrained')
        gs=fig.add_gridspec(2,2,height_ratios=[2,1])
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
        vmax=float(np.nanpercentile(amp,99))
        im=a.pcolormesh(x,z,amp,cmap='viridis',vmin=0,vmax=vmax,shading='nearest')
        fig.colorbar(im,ax=a,label='Complex spatial-mode magnitude (cm/s)',extend='max')
        im=b.pcolormesh(x,z,phase,cmap='twilight_shifted',vmin=-180,vmax=180,shading='nearest')
        fig.colorbar(im,ax=b,label='Spatial lag phase −arg(mode), degrees')
        for ax in [a,b]:
            ax.set(xlim=(130,285),ylim=(3000,0),xlabel='Longitude (°E)',ylabel='Depth (m)')
            ax.axhline(300,color='magenta',ls='--',lw=.7)
            ax.axhline(1900,color='gray',ls=':',lw=.7)
            ax.set_facecolor('.85')
        a.set_title('Spatial magnitude; not local U standard deviation')
        b.set_title('Phase masked below 5% of mode maximum magnitude')
        c.plot(time,tc[:,mode].real,label='Real')
        c.plot(time,tc[:,mode].imag,label='Imaginary')
        c.plot(time,abs(tc[:,mode]),color='black',ls=':',label='Magnitude')
        c.axhline(0,color='gray',lw=.7);c.set_ylabel('Dimensionless time coefficient');c.legend(ncol=3)
        label='Equal-point weights' if tag=='equal_points' else 'RF layer-thickness weights'
        fig.suptitle(f'CTRL 261016_ctrl1 | CEOF{mode+1}: {fractions[mode]*100:.2f}% | {label}\nEquatorial ±0.5° | 2–8 years | July 1999–June 2021',fontsize=13)
        fig.savefig(FIG/f'C0{mode+1}_{tag}.png',dpi=180);plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'source_sections.npz') as a:
        model=a['model'];times=pd.DatetimeIndex(a['time']);kk=a['kk']
    with xr.open_dataset(SOURCE/'section_statistics.nc') as d:
        x=d.longitude.values;z=d.depth.values
    mask=np.isfinite(model).all(axis=0)
    filtered=filt(model,times.month.to_numpy())
    analytic=np.full(filtered.shape,np.nan+1j*np.nan)
    analytic[:,mask]=hilbert(filtered[:,mask],axis=0)
    analytic=analytic[36:-36];time=times[36:-36]
    mask&=np.std(filtered[36:-36],axis=0)>1e-12
    rf=-np.fromfile(ROOT/'output/261016_ctrl1/RF.data',dtype='>f4').astype(float)
    dz=np.maximum(0,np.minimum(rf[kk+1],3000)-rf[kk])
    rows=[];checks={}
    for tag,w in [('equal_points',np.ones(mask.shape)),('RF_thickness',np.broadcast_to(dz[:,None],mask.shape))]:
        sm,tc,fractions,check=decompose(analytic,mask,w)
        checks[tag]=check
        for i in range(4):rows.append(dict(weighting=tag,mode=i+1,explained_variance=float(fractions[i]),cumulative_variance=float(fractions[:i+1].sum())))
        np.savez_compressed(OUT/f'ceof_{tag}.npz',spatial_mode=sm,time_coefficient=tc,explained_variance=fractions,time=time.values,depth=z,longitude=x,mask=mask,weights=w)
        xr.Dataset(dict(spatial_real=(('mode','depth','longitude'),sm.real),spatial_imag=(('mode','depth','longitude'),sm.imag),time_real=(('time','mode'),tc.real),time_imag=(('time','mode'),tc.imag),explained_variance=(('mode',),fractions[:4])),coords=dict(mode=np.arange(1,5),depth=z,longitude=x,time=time),attrs=dict(experiment='261016_ctrl1',weighting=tag,spatial_units='m/s',time_coefficient_units='dimensionless',reconstruction='Real(time_coefficient @ spatial_mode)',phase='spatial lag = -arg(spatial_mode); gauge TC[0] real positive',analytic_processing='Hilbert on full filtered support before 36-month trim; subtract complex temporal mean after trim',latitude='±0.5 degree wet-point average',filter='Own monthly climatology removed; fourth-order zero-phase 2–8 year bandpass',source_period='1996-07 to 2024-06',statistics_period='1999-07 to 2021-06')).to_netcdf(OUT/f'ceof_{tag}.nc')
        plot_modes(sm,tc,fractions,x,z,time,tag)
    table=pd.DataFrame(rows);table.to_csv(OUT/'explained_variance.csv',index=False)
    checks.update(source_model_wet_points=int(mask.sum()),time_samples=len(time),source_sha256=hashlib.sha256((SOURCE/'source_sections.npz').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    print(table.to_string(index=False),flush=True)
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
