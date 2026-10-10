"""Deep-section CEOF and Hovmoller diagnostics from compare_equatorial_u outputs."""
import argparse
import json
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from mitkit.paths import project_root


def decompose(da, weight, valid):
    x=da.values.reshape(da.sizes['time'],-1)[:,valid]
    x=x-x.mean(axis=0)
    analytic=hilbert(x,axis=0)
    weighted=analytic*np.sqrt(weight[valid])
    eig,vec=np.linalg.eigh(weighted@weighted.conj().T)
    order=np.argsort(eig)[::-1]; eig=np.maximum(eig[order],0)
    tc=vec[:,order[:3]]*np.sqrt(len(x))
    sm=tc.conj().T@analytic/len(x)
    return tc,sm,eig[:3]/eig.sum()


def analyze(exp):
    dest=project_root()/'fig'/f'{exp}_cmems_comparison'
    ds=xr.load_dataset(dest/'interannual_sections.nc').sel(depth=slice(1000,3000),lon=slice(140,280))
    z=ds.depth.values
    edges=np.r_[1000,(z[1:]+z[:-1])/2,3000]
    dz=xr.DataArray(np.diff(edges),dims='depth',coords={'depth':ds.depth})
    weight=np.broadcast_to(np.diff(edges)[:,None],(len(z),ds.sizes['lon'])).ravel().copy()
    weight/=np.sum(weight)
    valid=np.isfinite(ds.model.values).all(axis=0).ravel() & np.isfinite(ds.cmems.values).all(axis=0).ravel()
    mt,ms,mv=decompose(ds.model,weight,valid)
    ct,cs,cv=decompose(ds.cmems,weight,valid)
    similarity=mt.conj().T@ct/len(mt)
    # Rotate model phase against CMEMS; amplitude stays in physical units.
    rotation=np.exp(1j*np.angle(similarity[0,0]))
    mt[:,0]*=rotation; ms[0]*=rotation.conjugate()
    fig,axs=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    ampmax=float(max(np.max(np.abs(ms[0])),np.max(np.abs(cs[0])))*100)
    for j,(sm,tc,var,label) in enumerate([(cs,ct,cv,'CMEMS'),(ms,mt,mv,'Model')]):
        field=np.full(len(weight),np.nan+1j*np.nan); field[valid]=sm[0]; field=field.reshape(len(z),-1)
        im=axs[0,j].pcolormesh(ds.lon,z,np.abs(field)*100,cmap='YlOrRd',vmin=0,vmax=ampmax,shading='auto')
        axs[0,j].set_title(f'{label} CEOF1 amplitude ({var[0]*100:.1f}% variance)')
        fig.colorbar(im,ax=axs[0,j],label='cm/s')
        phase=np.angle(field); phase[np.abs(field)<np.nanmax(np.abs(field))*.1]=np.nan
        im=axs[1,j].pcolormesh(ds.lon,z,phase,cmap='twilight',vmin=-np.pi,vmax=np.pi,shading='auto')
        axs[1,j].set_title(f'{label} spatial phase (10% amplitude mask)'); fig.colorbar(im,ax=axs[1,j],label='rad')
        for i in (0,1): axs[i,j].set(ylim=(3000,1000),xlabel='Longitude (°E)',ylabel='Depth (m)')
        axs[2,j].plot(ds.time,tc[:,0].real,label='Real'); axs[2,j].plot(ds.time,tc[:,0].imag,label='Imaginary')
        axs[2,j].set_title('Normalized time coefficient'); axs[2,j].legend()
    fig.savefig(dest/'deep_ceof.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,9),sharex=True,sharey=True,constrained_layout=True)
    a=ds.model.weighted(dz).mean('depth'); b=ds.cmems.weighted(dz).mean('depth')
    limit=float(max(np.abs(a).quantile(.99),np.abs(b).quantile(.99)))*100
    for ax,field,label in zip(axs,[b,a,a-b],['CMEMS','Model','Model minus CMEMS']):
        im=ax.pcolormesh(ds.lon,ds.time,field.transpose('time','lon')*100,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto')
        ax.set_title(label); ax.set_xlabel('Longitude (°E)')
    axs[0].set_ylabel('Time')
    fig.suptitle('1000–3000 m thickness-weighted U, 2–8 years')
    fig.colorbar(im,ax=axs,label='cm/s')
    fig.savefig(dest/'deep_hovmoller.png',dpi=180); plt.close(fig)
    summary={'model_explained_variance':mv.tolist(),'cmems_explained_variance':cv.tolist(),'complex_temporal_similarity_abs':np.abs(similarity).tolist(),'phase_definition':'arg(spatial mode); model CEOF1 rotated to align with CMEMS CEOF1','domain':'140–280 E, 1000–3000 m','weighting':'layer thickness, uniform longitude; common complete wet cells','caveat':'Independent modes can reorder or mix; compare full similarity matrix. Hilbert transform is applied after filter endpoint trimming.'}
    (dest/'deep_modes.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp',default='260821_ctrl')
    analyze(parser.parse_args().exp)
