"""Common N.mat velocity modes and n=0..5 equatorial-section comparison."""
import argparse,json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_western_deep import ROOT,bandpass,weights,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp',nargs='+',default=['261008_ck_0p10','261008_ck_0p07','261010_a4_40e9'])
    p.add_argument('--cutoff',type=int,default=5)
    p.add_argument('--out',default='fig/low_vertical_modes_comparison')
    args=p.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    ds=xr.align(*[xr.load_dataset(ROOT/'fig'/f'{e}_cmems_comparison/sections.nc') for e in args.exp],join='inner')
    ref=loadmat(ROOT/'analysis/ceof/N.mat');zn=ref['z'].ravel();nf=ref['n'].ravel()
    z=ds[0].depth.values;H=float(zn.max())
    edges=np.r_[0,(z[:-1]+z[1:])/2,H];w=np.diff(edges)
    if np.any(w<=0) or np.any(nf<=0):raise ValueError('Invalid depths or buoyancy frequency')
    a=np.zeros((len(z),len(z)))
    n2=np.interp(edges[1:-1],zn,nf)**2
    for k,g in enumerate(1/(n2*np.diff(z))):
        a[k,k]+=g;a[k+1,k+1]+=g;a[k,k+1]-=g;a[k+1,k]-=g
    ev,phi=eigh(a,np.diag(w));phi*=np.sqrt(H)
    phi*=np.where(phi[0]<0,-1,1)
    speed=np.r_[np.nan,1/np.sqrt(ev[1:])]
    k=args.cutoff+1
    if k>=len(z):raise ValueError('Too many retained modes')
    good=ds[0].cmems.notnull().all('time').all('depth')
    for d in ds:good=good&d.model.notnull().all('time').all('depth')
    if not good.any():raise ValueError('No complete common profiles')
    raw=[ds[0].cmems]+[d.model for d in ds];names=['CMEMS']+args.exp
    lows=[];coeffs=[];rows=[];profiles=[]
    for name,u in zip(names,raw):
        x=u.transpose('time','depth','lon').where(good).values
        coef=np.einsum('zm,tzx,z->tmx',phi,x,w)/H
        low=np.einsum('zm,tmx->tzx',phi[:,:k],coef[:,:k])
        da=xr.DataArray(low,dims=('time','depth','lon'),coords=u.transpose('time','depth','lon').coords)
        ca=xr.DataArray(coef[:,:k],dims=('time','vertical_mode','lon'),coords={'time':u.time,'vertical_mode':np.arange(k),'lon':u.lon})
        lows.append(da);coeffs.append(ca)
        xr.Dataset({'low_u':da,'mode_coefficient':ca,'original_u':u.where(good),'high_residual_u':u.where(good)-da}).to_netcdf(out/f'{name}_reconstruction.nc')
    xr.Dataset({'velocity_mode':(('depth','vertical_mode'),phi[:,:k]),'quadrature_weight':('depth',w),'phase_speed':('vertical_mode',speed[:k])},coords={'depth':z,'vertical_mode':np.arange(k)},attrs={'reference':'analysis/ceof/N.mat','N_units':'s^-1; n treated as N, squared for N2','boundary':'rigid-lid Neumann velocity-mode problem on 0–3001 m','normalization':'integral phi_i phi_j dz / H = delta_ij'}).to_netcdf(out/'basis.nc')
    for west,east in [(130,150),(140,170),(170,220),(220,280)]:
        for lo,hi in [(0,300),(300,1000),(1000,3000),(1000,1500),(1500,2000),(2000,3000)]:
            def avg(v):return v.sel(lon=slice(west,east)).weighted(weights(v,lo,hi)).mean(('depth','lon'))
            series=[avg(v) for v in lows]
            for trim in [24,36,48]:
                signals=[bandpass(v,trim) for v in series]
                for i,e in enumerate(args.exp,1):
                    rows.append(dict(experiment=e,trim=trim,longitude=f'{west}-{east}',depth=f'{lo}-{hi}',mean_bias_cm_s=float((series[i]-series[0]).mean()*100),**metrics(signals[i].values,signals[0].values)))
    pd.DataFrame(rows).to_csv(out/'regional_metrics.csv',index=False)
    fs=[bandpass(v,36) for v in lows];cs=[bandpass(v,36) for v in coeffs]
    for i,e in enumerate(args.exp,1):
        for n in range(k):
            for west,east in [(130,150),(170,220),(220,280)]:
                m=cs[i].sel(vertical_mode=n,lon=slice(west,east)).mean('lon');c=cs[0].sel(vertical_mode=n,lon=slice(west,east)).mean('lon')
                profiles.append(dict(experiment=e,vertical_mode=n,longitude=f'{west}-{east}',**metrics(m.values,c.values)))
    pd.DataFrame(profiles).to_csv(out/'mode_metrics.csv',index=False)
    for lo,hi,tag in [(0,1000,'upper_middle'),(1000,3000,'deep')]:
        fig,axes=plt.subplots(3,len(names),figsize=(4*len(names),10),sharex=True,sharey=True,constrained_layout=True)
        arrays=[[v.mean('time')*100 for v in lows],[v.std('time')*100 for v in fs],[xr.corr(v,fs[0],dim='time') for v in fs]]
        for r,vals in enumerate(arrays):
            vals=[v.sel(depth=slice(lo,hi)) for v in vals]
            vmax=1 if r==2 else max(float(np.nanquantile(np.abs(v),.98)) for v in vals)
            for ax,v,name in zip(axes[r],vals,names):
                im=ax.pcolormesh(v.lon,v.depth,v,shading='auto',cmap='viridis' if r==1 else 'RdBu_r',vmin=0 if r==1 else -vmax,vmax=vmax)
                ax.set(title=name,ylim=(hi,lo),xlabel='Longitude (°E)')
            fig.colorbar(im,ax=list(axes[r]),label=['Low-mode mean U (cm/s)','2–8 yr std (cm/s)','Correlation with CMEMS'][r])
        fig.savefig(out/f'{tag}_low_modes.png',dpi=160);plt.close(fig)
    info=dict(experiments=args.exp,retained_modes=list(range(k)),complete_longitude_count=int(good.sum()),total_longitude_count=len(good),reference_depth_m=H,phase_speeds_m_s=[None]+speed[1:k].tolist(),orthogonality_error=float(np.max(np.abs(phi.T@np.diag(w)@phi/H-np.eye(len(z))))),notes=['Common reference buoyancy frequency and rigid-lid velocity modes; not CEOF.','Only full-depth common wet profiles retained; gaps are not filled or projected as zero.','Analysis restricted to N.mat reference depth; not full ocean-column modes.','2–8 yr comparison; 18-month variability is outside the research scope.','No observation transport or reproduction of the reported 94% is attempted.'])
    (out/'summary.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
