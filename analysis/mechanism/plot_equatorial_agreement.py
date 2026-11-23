"""Joint spatial view of interannual timing and amplitude; reuse audited sections."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research/mechanism/2026-11-17_equatorial_agreement'
FIG = ROOT/'fig/261117_equatorial_agreement'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source = ROOT/'research/mechanism/2026-11-16_depth_amplitude/section_statistics.nc'
    with xr.open_dataset(source) as ds:
        ds = ds.load()
    x, z = ds.longitude.values, ds.depth.values
    r, q = ds.correlation.values, ds.std_ratio.values
    assert np.nanmax(abs(r)) <= 1 + 1e-12
    assert np.nanmin(q) >= 0
    # Validate the saved Pearson map independently from filtered source arrays.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from diagnose_depth_amplitude import filt
    with np.load(source.parent/'source_sections.npz') as a:
        months = pd.DatetimeIndex(a['time']).month.to_numpy()
        m = filt(a['model'], months)[36:-36]
        g = filt(a['gl12'], months)[36:-36]
    mask = np.isfinite(r)
    m, g = m[:, mask], g[:, mask]
    m -= m.mean(axis=0)
    g -= g.mean(axis=0)
    fresh = np.sum(m*g, axis=0)/np.sqrt(np.sum(m*m, axis=0)*np.sum(g*g, axis=0))
    err = float(np.max(abs(fresh-r[mask])))
    assert err < 1e-10
    for limit, tag in [(3000, 'full'), (700, 'upper')]:
        fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True, layout='constrained')
        for ax, key, title in zip(axs[0], ['gl12_interannual_std', 'model_interannual_std'], ['(a) GL12 amplitude', '(b) CTRL amplitude']):
            im = ax.pcolormesh(x, z, np.where(mask, ds[key].values*100, np.nan), cmap='viridis', vmin=0, vmax=10, shading='nearest')
            ax.set_title(title)
        fig.colorbar(im, ax=list(axs[0]), label='2–8 year U standard deviation (cm/s); values >10 saturated', extend='max', shrink=.85)
        im = axs[1,0].pcolormesh(x, z, r, cmap='RdBu_r', vmin=-1, vmax=1, shading='nearest')
        cs = axs[1,0].contour(x, z, q, levels=[.5, 1, 1.5], colors=['black'], linewidths=.65)
        axs[1,0].clabel(cs, fmt={.5:'A=0.5', 1:'A=1', 1.5:'A=1.5'}, fontsize=7)
        axs[1,0].set_title('(c) Correlation r; contours: amplitude ratio A')
        fig.colorbar(im, ax=axs[1,0], label='Zero-lag Pearson r', shrink=.85)
        im = axs[1,1].pcolormesh(x, z, q, cmap='BrBG_r', norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2), shading='nearest')
        cs = axs[1,1].contour(x, z, r, levels=[0, .5, .8], colors='black', linewidths=.65)
        axs[1,1].clabel(cs, fmt=lambda v:f'r={v:g}', fontsize=7)
        axs[1,1].set_title('(d) Amplitude ratio A; contours: correlation r')
        fig.colorbar(im, ax=axs[1,1], label='A = CTRL std / GL12 std; >2 saturated', extend='max', shrink=.85)
        for ax in axs.flat:
            ax.set_facecolor('.85')
            ax.set(ylim=(limit, 0), xlim=(130,285), xlabel='Longitude (°E)', ylabel='Depth (m)')
            ax.axhline(300, color='magenta', ls='--', lw=.8)
            if limit>1900:
                ax.axhline(1900, color='gray', ls=':', lw=.8)
            ax.axvline(142, color='gray', ls=':', lw=.8)
        fig.suptitle('Equatorial ±0.5° wet-point mean U | 2–8 years | July 1999–June 2021\nSame common grid and wet mask; gray = unavailable; magenta = 300 m')
        fig.savefig(FIG/f'A01_agreement_{tag}.png', dpi=190)
        plt.close(fig)
    # Descriptive depth/longitude-weighted summaries, not independent samples.
    folder=ROOT/'output/261016_ctrl1'
    rf=-np.fromfile(folder/'RF.data', dtype='>f4').astype(float)
    with np.load(source.parent/'source_sections.npz') as a:
        kk=a['kk']
    rows=[]
    for name,lo,hi in [('west',130,150),('central',170,220),('east',220,255)]:
        for top,bot in [(100,300),(300,500),(500,1000),(1000,1900),(1900,3000)]:
            dz=np.maximum(0,np.minimum(rf[kk+1],bot)-np.maximum(rf[kk],top))
            w=dz[:,None]*((x>=lo)&(x<hi))[None,:]*mask
            def avg(v):
                return float(np.sum(np.where(mask,v,0)*w)/w.sum())
            rows.append(dict(region=name,top_m=top,bottom_m=bot,weighted_mean_local_r=avg(r),fraction_r_ge_05=avg(r>=.5),fraction_r_negative=avg(r<0),fraction_r_ge_05_and_A_lt_075=avg((r>=.5)&(q<.75))))
    pd.DataFrame(rows).to_csv(OUT/'regional_agreement.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),independent_correlation_max_abs_error=err,common_points=int(mask.sum()),period='1999-07 to 2021-06',filter='Own monthly climatology removed; fourth-order zero-phase 2–8 year bandpass on 1996-07 to 2024-06, trim 36 months each end',summary_weights='RF thickness clipped to band, uniform longitude; descriptive, not significance tests',limits=['Correlation is zero lag and is not propagation direction or causality','No nominal monthly-sample significance tests for strongly autocorrelated series','No amplitude-ratio masking threshold chosen from observed agreement; inspect absolute amplitudes in panels a,b']),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print('Correlation recheck max error:',err)

if __name__=='__main__':
    main()
