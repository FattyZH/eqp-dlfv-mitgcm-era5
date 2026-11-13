"""Area-integrated 142E channel transport below a fixed depth.

Native U-face geometry for MITgcm; published GLORYS uo cells, static bathymetry
and e3t for the reference. Exact cell-overlap integration for common wet area.
Missing deep reference levels are never filled or called full-bottom transport.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix, vstack
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/common'))
from analysis_common import meta, sha, dump
ALGORITHM='fixed_depth_channel_v2_bracketing_face_flux'


def bounds_from_centers(x):
    """Latitude cell bounds for a regular published cell-center coordinate."""
    x=np.asarray(x,float)
    if not np.all(np.diff(x)>0):raise ValueError('Coordinates must increase')
    e=np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
    return e[:-1],e[1:]


def coastal_cells(lat,lower,upper,surface_wet,north):
    """Connected surface ocean south of north boundary, stopping at island land."""
    candidates=np.flatnonzero((lower<north)&(upper>-5))
    if not len(candidates):raise ValueError('North boundary outside data')
    j=candidates[-1]
    if not surface_wet[j]:raise ValueError('North boundary must be ocean')
    stop=j
    while stop>=0 and surface_wet[stop] and upper[stop]>-5:stop-=1
    if stop<0 or upper[stop]<=-5:raise ValueError('No New Guinea coastline before 5S')
    keep=np.arange(stop+1,j+1)
    return keep,float(lower[keep[0]])


def clipped_thickness(top,bottom,upper_depth):
    return np.maximum(0,bottom-np.maximum(top,upper_depth))


def common_operator(m,r,upper_depth):
    """Sparse exact overlap areas (m cells x reference cells), using U-face DYG.

    Velocities are interpreted as piecewise constant native cell values. All
    overlap weights use model meridional metric to give both series SAME area.
    """
    if m['top'].ndim==3:
        blocks=[]
        for f,w in enumerate(m['face_weights']):
            part={k:(v[f] if k in ['top','bottom','metric_per_degree'] else v) for k,v in m.items()}
            blocks.append(common_operator(part,r,upper_depth)*w)
        return vstack(blocks,format='csr')
    mz,my=m['top'].shape;rz,ry=r['top'].shape;rows=[];cols=[];values=[]
    for j in range(my):
        for l in range(ry):
            overlap=max(0,min(m['north'][j],r['north'][l])-max(m['south'][j],r['south'][l]))
            if overlap<=0:continue
            dy=m['metric_per_degree'][j]*overlap
            dz=np.maximum(0,np.minimum(m['bottom'][:,j,None],r['bottom'][None,:,l])-np.maximum(np.maximum(m['top'][:,j,None],r['top'][None,:,l]),upper_depth))
            ii,kk=np.nonzero(dz>0)
            rows.extend((ii*my+j).tolist());cols.extend((kk*ry+l).tolist());values.extend((dz[ii,kk]*dy).tolist())
    return coo_matrix((values,(rows,cols)),shape=(mz*my,rz*ry)).tocsr()


def integrate(velocity,area):
    valid=area>0
    if not np.isfinite(velocity[:,valid]).all():raise ValueError('Missing velocity within integration area; do not replace with zero')
    a=velocity[:,valid];w=area[valid]
    net=a@w/1e6;east=np.maximum(a,0)@w/1e6;west=np.minimum(a,0)@w/1e6
    if not np.allclose(net,east+west,rtol=1e-12,atol=1e-10):raise AssertionError('Directional transport closure')
    return net,east,west


def bandpass(values,months):
    a=np.asarray(values,float).copy()
    for mon in range(1,13):a[months==mon]-=a[months==mon].mean()
    return sosfiltfilt(butter(4,[1/8,1/2],fs=12,btype='bandpass',output='sos'),a)


def model_geometry(folder,north):
    nx,ny,nz=720,192,40
    def rd(name,shape):return np.fromfile(folder/(name+'.data'),dtype='>f4').reshape(shape).astype(float)
    x=rd('XG',(ny,nx))[0];hi=int(np.searchsorted(x,142));lo=hi-1
    if hi>=len(x) or lo<0:raise ValueError('142E outside model U grid')
    faces=np.array([lo,hi]);weights=np.array([x[hi]-142,142-x[lo]])/(x[hi]-x[lo])
    lat=rd('YC',(ny,nx))[:,lo];ys=rd('YG',(ny,nx))[:,lo];yn=np.r_[ys[1:],ys[-1]+ys[-1]-ys[-2]]
    hw=np.moveaxis(rd('hFacW',(nz,ny,nx))[:,:,faces],-1,0);rf=-rd('RF',(nz+1,));dy=rd('DYG',(ny,nx))[:,faces].T
    js,coast=coastal_cells(lat,ys,yn,(hw[:,0]>0).any(axis=0),north)
    top=np.broadcast_to(rf[None,:-1,None],(len(faces),nz,len(js))).copy();bottom=top+np.diff(rf)[None,:,None]*hw[:,:,js]
    for f in range(len(faces)):
        for j in js:
            active=np.flatnonzero(hw[f,:,j]>0)
            if len(active) and not np.array_equal(active,np.arange(active[-1]+1)):raise ValueError('Noncontiguous model wet column')
            if len(active)>1 and not np.allclose(hw[f,active[:-1],j],1,atol=1e-6):raise ValueError('Interior partial cells require explicit geometry')
    return dict(lon=142.,lat=lat[js],south=ys[js],north=np.minimum(yn[js],north),metric_per_degree=dy[:,js]/(yn[js]-ys[js])[None],top=top,bottom=bottom,wet=hw[:,:,js]>0,depth=rd('RC',(nz,))*-1,bathy=(np.diff(rf)[None,:,None]*hw[:,:,js]).sum(axis=1),js=js,faces=faces,face_longitudes=x[faces],face_weights=weights,coast=coast)



def reference_geometry(ds,coordinate_file,bathy_file,north):
    x=ds.longitude.values.astype(float);i=int(np.argmin(abs(x-142)))
    if abs(x[i]-142)>1e-5:raise ValueError('Reference must contain 142E; horizontal interpolation needs a separate wet-geometry definition')
    lat=ds.latitude.values.astype(float);ys,yn=bounds_from_centers(lat);z=ds.depth.values.astype(float)
    with xr.open_dataset(coordinate_file) as co, xr.open_dataset(bathy_file) as ba:
        c=co.sel(longitude=142,method='nearest').sel(latitude=xr.DataArray(lat,dims='latitude'),method='nearest').load()
        b=ba.sel(longitude=142,method='nearest').sel(latitude=xr.DataArray(lat,dims='latitude'),method='nearest').load()
        if np.max(abs(c.latitude.values-lat))>1e-4 or np.max(abs(b.latitude.values-lat))>1e-4:raise ValueError('Static geometry grid mismatch')
        nz=len(z)
        if not np.allclose(z,co.depth.values[:nz],atol=.01):raise ValueError('Reference depths are not leading native levels; supply matching thicknesses')
        dz=co.e3t.values.astype(float);edges=np.r_[0,np.cumsum(dz)];mask=b['mask'].values.astype(bool);bath=b.deptho.values.astype(float)
        js,coast=coastal_cells(lat,ys,yn,mask[0],north)
        top=np.broadcast_to(edges[:nz,None],(nz,len(js))).copy();bottom=np.minimum(edges[1:nz+1,None],bath[js][None]);wet=mask[:nz,js];bottom=np.where(wet,np.maximum(bottom,top),top)
        # Deep missing area relative to bathymetry and all native wet cells.
        ft=np.broadcast_to(edges[:-1,None],(len(dz),len(js))).copy();fb=np.where(mask[:,js],np.maximum(np.minimum(edges[1:,None],bath[js][None]),ft),ft)
        metric=c.e2t.values.astype(float)[js]/(yn[js]-ys[js])
    return dict(lon=142.,lat=lat[js],south=ys[js],north=np.minimum(yn[js],north),metric_per_degree=metric,top=top,bottom=bottom,wet=wet,depth=z,bathy=bath[js],js=js,i=i,coast=coast,full_top=ft,full_bottom=fb,available_layer_bottom_m=float(edges[nz]))


def fingerprint(files,parameters):
    entries=[(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps(dict(files=entries,parameters=parameters),sort_keys=True).encode()).hexdigest()


def extract(args,m,r,times,dest):
    folder=ROOT/'output'/args.exp;wanted=set(times);sources=[]
    for path in sorted(folder.glob('diag3d.*.meta')):
        nr,fields,t=meta(path)
        if t in wanted:sources.append((path,nr,fields,t))
    if not pd.DatetimeIndex([s[3] for s in sources]).equals(times):raise ValueError('Model/reference months are not matched')
    fs=[args.reference,args.coordinates,args.bathymetry]+[folder/(name+'.data') for name in ['XG','YG','YC','DYG','RF','RC','hFacW']]
    for p,_,_,_ in sources:fs.extend([p,p.with_suffix('.data')])
    sig=fingerprint(fs,dict(algorithm=ALGORITHM,exp=args.exp,north=args.north,longitude=142))
    cache=dest/'source_sections.nc';tag=dest/'source_signature.json'
    if cache.exists() and tag.exists() and json.loads(tag.read_text())['signature']==sig:
        saved=xr.load_dataset(cache);print('Reusing source section cache',flush=True);return saved.model_u.values,saved.reference_u.values,sig
    def read(item):
        p,nr,fields,t=item;a=np.memmap(p.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,40,192,720));v=np.moveaxis(np.array(a[fields.index('UVEL'),:,m['js'][0]:m['js'][-1]+1,:],float)[:,:,m['faces']],-1,0);v[~m['wet']]=np.nan;return v
    result=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n,v in enumerate(pool.map(read,sources)):
            result.append(v)
            if (n+1)%60==0:print('Model section months',n+1,flush=True)
    model=np.stack(result)
    with xr.open_dataset(args.reference) as ds:
        ref=ds.uo.sel(time=slice(times[0],times[-1]+pd.offsets.MonthEnd(0))).isel(longitude=r['i'],latitude=r['js']).transpose('time','depth','latitude').load().values.astype(float)
    if len(ref)!=len(times):raise ValueError('Reference selection mismatch')
    print('Loaded reference section',ref.shape,flush=True)
    xr.Dataset(dict(model_u=(('time','model_face','model_depth','model_latitude'),model),reference_u=(('time','reference_depth','reference_latitude'),ref)),coords=dict(time=times,model_face=m['face_longitudes'],model_depth=m['depth'],model_latitude=m['lat'],reference_depth=r['depth'],reference_latitude=r['lat']),attrs=dict(longitude=142.,velocity_units='m/s',algorithm=ALGORITHM)).to_netcdf(cache)
    dump(tag,dict(signature=sig,algorithm=ALGORITHM));return model,ref,sig


def metrics(table,times,dest):
    monthly=times.month.to_numpy();f={k:bandpass(v,monthly) for k,v in table.items() if k.endswith('_net_Sv')};rows=[]
    for trim in [24,36,48]:
        m=f['model_common_net_Sv'][trim:-trim];r=f['reference_common_net_Sv'][trim:-trim]
        rows.append(dict(trim_months=trim,first_month=str(times[trim].date()),last_month=str(times[-trim-1].date()),correlation=float(np.corrcoef(m,r)[0,1]),model_std_Sv=float(np.std(m)),reference_std_Sv=float(np.std(r)),std_ratio=float(np.std(m)/np.std(r)),anomaly_rmse_Sv=float(np.sqrt(np.mean((m-r)**2))),mean_bias_full_period_Sv=float(np.mean(table['model_common_net_Sv']-table['reference_common_net_Sv']))))
    pd.DataFrame(rows).to_csv(dest/'comparison_metrics.csv',index=False)
    pd.DataFrame(f,index=times).rename_axis('time').to_csv(dest/'filtered_transports.csv')
    for key,v in table.items():
        if key.endswith('_net_Sv'):
            print(key,'mean',np.mean(v),'std',np.std(v),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return f,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp',default='261016_ctrl');parser.add_argument('--reference',type=Path,default=ROOT/'data/cmems/gl12_uvt_9607-2406_fd.nc');parser.add_argument('--coordinates',type=Path,default=Path('/public/home/zhanghang/data/GLO-MFC_001_030_coordinates.nc'));parser.add_argument('--bathymetry',type=Path,default=Path('/public/home/zhanghang/data/GLO-MFC_001_030_mask_bathy.nc'));parser.add_argument('--upper-depth',type=float,default=1900);parser.add_argument('--north',type=float,default=2);parser.add_argument('--allow-incomplete-reference',action='store_true',help='Compatibility flag; available-level comparison is already the default');parser.add_argument('--require-full-depth',action='store_true',help='Stop if any bathymetric wet area lies below available reference levels');parser.add_argument('--workers',type=int,default=int(os.environ.get('EQP_IO_WORKERS','16')));parser.add_argument('--output',type=Path,default=None);args=parser.parse_args()
    tag=('ctrl' if args.exp=='261016_ctrl' else args.exp);suffix='' if args.north==2 else f'_{args.north:g}N';dest=args.output or ROOT/'research/transport'/f'2026-11-13_{args.upper_depth:g}m_{tag}{suffix}';dest.mkdir(parents=True,exist_ok=True);plot=ROOT/'fig'/('261113_transport_'+args.exp+('' if args.upper_depth==1900 and args.north==2 else f'_{args.upper_depth:g}m_{args.north:g}N'));plot.mkdir(parents=True,exist_ok=True)
    m=model_geometry(ROOT/'output'/args.exp,args.north)
    with xr.open_dataset(args.reference) as ds:
        if ds.uo.attrs.get('units')!='m s-1':raise ValueError('Unexpected velocity units')
        times=pd.DatetimeIndex(ds.time.values).to_period('M').to_timestamp();r=reference_geometry(ds,args.coordinates,args.bathymetry,args.north)
    areas={}
    for key,g in [('model',m),('reference_available',r)]:areas[key]=clipped_thickness(g['top'],g['bottom'],args.upper_depth)*(g['metric_per_degree']*(g['north']-g['south']))[:,None,:] * g['face_weights'][:,None,None] if key=='model' else clipped_thickness(g['top'],g['bottom'],args.upper_depth)*(g['metric_per_degree']*(g['north']-g['south']))[None]
    ref_full=clipped_thickness(r['full_top'],r['full_bottom'],args.upper_depth)*(r['metric_per_degree']*(r['north']-r['south']))[None];missing=float(ref_full.sum()-areas['reference_available'].sum());fraction=missing/ref_full.sum();op=common_operator(m,r,args.upper_depth);ma=np.array(op.sum(axis=1)).reshape(m['top'].shape);ra=np.array(op.sum(axis=0)).reshape(r['top'].shape)
    if not np.isclose(ma.sum(),ra.sum(),rtol=1e-12):raise AssertionError('Common area conservation')
    if ma.sum()<=0:raise ValueError('Empty common section')
    geometry=dict(definition=dict(longitude_E=142,north_latitude_N=args.north,upper_depth_m=args.upper_depth,lower='local seafloor',positive='eastward',units='Sv',model_face_longitudes=m['face_longitudes'].tolist(),model_face_weights=m['face_weights'].tolist(),model_meridian_method='Linear interpolation of integrated flux/area between bracketing U faces; preserve each face hFacW geometry before weighting',north_boundary='exact fractional cell clipping',model_coast_latitude=m['coast'],reference_coast_latitude=r['coast']),model_native_area_km2=areas['model'].sum()/1e6,reference_full_area_km2=ref_full.sum()/1e6,reference_available_area_km2=areas['reference_available'].sum()/1e6,missing_reference_area_km2=missing/1e6,missing_reference_area_fraction=fraction,common_area_km2=ma.sum()/1e6,reference_complete=bool(fraction<1e-10),reference_available_layer_bottom_m=r['available_layer_bottom_m'],model_max_bottom_m=float(m['bathy'].max()),reference_max_bottom_m=float(r['bathy'].max()),common_metric='Model DYG scaled by exact latitude overlap; piecewise constant native velocities; sparse exact native cell intersections',reference_geometry='Static e3t cumulative interfaces, deptho bottom clipping, published mask and e2t; no midpoint extrapolation of last available depth')
    dump(dest/'geometry.json',geometry);print(json.dumps(geometry,indent=2),flush=True)
    if fraction>1e-10 and args.require_full_depth:raise SystemExit('Reference incomplete below available layers. Supply full-depth uo or omit --require-full-depth for explicitly labelled available/common-area comparison.')
    model,ref,sig=extract(args,m,r,times,dest)
    table={}
    for name,a,area in [('model_native',model,areas['model']),('reference_available',ref,areas['reference_available']),('model_common',model,ma),('reference_common',ref,ra)]:
        for direction,v in zip(['net','east','west'],integrate(a,area)):table[name+'_'+direction+'_Sv']=v
    pd.DataFrame(table,index=times).rename_axis('time').to_csv(dest/'transports.csv');f,rows=metrics(table,times,dest)
    # Depth-bound sensitivity uses source velocities without new source reads.
    sensitivity=[]
    for depth in [1800,1900,2000]:
        oo=common_operator(m,r,depth);a=np.asarray(oo.sum(axis=1)).reshape(ma.shape);b=np.asarray(oo.sum(axis=0)).reshape(ra.shape);qm=integrate(model,a)[0];qr=integrate(ref,b)[0];fm=bandpass(qm,times.month.to_numpy())[36:-36];fr=bandpass(qr,times.month.to_numpy())[36:-36];sensitivity.append(dict(upper_depth_m=depth,common_area_km2=a.sum()/1e6,model_mean_Sv=qm.mean(),reference_mean_Sv=qr.mean(),model_std_Sv=fm.std(),reference_std_Sv=fr.std(),correlation=np.corrcoef(fm,fr)[0,1],std_ratio=fm.std()/fr.std()))
    pd.DataFrame(sensitivity).to_csv(dest/'upper_depth_sensitivity.csv',index=False)
    # Save geometry and transport per model latitude on exactly the same area.
    mr=(model.reshape(len(times),-1)*np.array(op.sum(axis=1)).ravel()[None]).reshape(model.shape)
    mr=np.where(ma[None]>0,mr,0).sum(axis=(1,2))/1e6
    rr=np.asarray(op@np.where(np.isfinite(ref),ref,0).reshape(len(times),-1).T).T.reshape(model.shape).sum(axis=(1,2))/1e6
    if not np.allclose(mr.sum(axis=1),table['model_common_net_Sv'],atol=1e-10):raise AssertionError('Latitude model closure')
    if not np.allclose(rr.sum(axis=1),table['reference_common_net_Sv'],atol=1e-10):raise AssertionError('Latitude reference closure')
    xr.Dataset(dict(model_native_area=(('model_face','model_depth','model_latitude'),areas['model']),model_common_area=(('model_face','model_depth','model_latitude'),ma),reference_available_area=(('reference_depth','reference_latitude'),areas['reference_available']),reference_common_area=(('reference_depth','reference_latitude'),ra),model_transport_by_latitude=(('time','model_latitude'),mr),reference_transport_by_model_latitude=(('time','model_latitude'),rr)),coords=dict(time=times,model_face=m['face_longitudes'],model_depth=m['depth'],model_latitude=m['lat'],reference_depth=r['depth'],reference_latitude=r['lat']),attrs=dict(area_units='m2',transport_units='Sv')).to_netcdf(dest/'geometry_and_latitude_transport.nc')
    checks=dict(common_area_conserved=bool(np.isclose(ma.sum(),ra.sum(),rtol=1e-12)),constant_1m_s_common_transport_Sv=float(ma.sum()/1e6),directional_closure=True,latitude_transport_closure=True,no_missing_values_in_positive_area=True,reference_complete=geometry['reference_complete'],reference_missing_not_filled=True,source_signature=sig)
    dump(dest/'validation.json',checks)
    fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for key,col in [('model_common','tab:orange'),('reference_common','black')]:
        axs[0].plot(times,table[key+'_net_Sv'],label=key,color=col);axs[1].plot(times[36:-36],f[key+'_net_Sv'][36:-36],label=key,color=col)
    for ax in axs:ax.axhline(0,color='gray',lw=.7);ax.set(ylabel='Eastward transport (Sv)');ax.legend()
    axs[0].set_title('Monthly net transport on identical common wet area');axs[1].set_title(f'2–8 year anomalies | r={rows[1]["correlation"]:.2f}, std ratio={rows[1]["std_ratio"]:.2f}');fig.suptitle(f'142 E, island to {args.north:g} N, below {args.upper_depth:g} m | reference missing area {fraction:.1%}');fig.savefig(plot/'Q01_transport_series.png',dpi=180);plt.close(fig)
    # Draw actual wet-cell rectangles: center-based pcolormesh would paint below
    # sloping bottoms and misrepresent partial cells. Show both native U faces.
    fig,axs=plt.subplots(1,3,figsize=(16,6),layout='constrained',sharey=True)
    fields=[(m,model[:,0],0,f'CTRL U face {m["face_longitudes"][0]:g} E'),(m,model[:,1],1,f'CTRL U face {m["face_longitudes"][1]:g} E'),(r,ref,None,'GLORYS 142 E: available cells')]
    for ax,(g,a,face,title) in zip(axs,fields):
        count=np.isfinite(a).sum(axis=0);mu=np.divide(np.nansum(a,axis=0),count,out=np.full(a.shape[1:],np.nan),where=count>0)
        top=g['top'] if face is None else g['top'][face];bottom=g['bottom'] if face is None else g['bottom'][face];bath=g['bathy'] if face is None else g['bathy'][face];polygons=[];colors=[]
        for k in range(len(g['depth'])):
            for j in range(len(g['lat'])):
                if bottom[k,j]>top[k,j] and np.isfinite(mu[k,j]):
                    polygons.append([(g['south'][j],top[k,j]),(g['north'][j],top[k,j]),(g['north'][j],bottom[k,j]),(g['south'][j],bottom[k,j])]);colors.append(mu[k,j]*100)
        im=PolyCollection(polygons,array=np.array(colors),cmap='RdBu_r',norm=Normalize(-10,10),edgecolors='none');ax.add_collection(im);ax.plot(g['lat'],bath,'k-',lw=1);ax.axhline(args.upper_depth,color='gray',ls='--');ax.set(title=title,xlabel='Latitude (N)',ylim=(4300,1400),xlim=(-3.2,args.north))
        if face is None:ax.axhline(r['available_layer_bottom_m'],color='red',ls=':',label='Available cell bottom');ax.legend(fontsize=8)
    axs[0].set_ylabel('Depth (m)');fig.colorbar(im,ax=axs,label='Mean eastward U (cm/s)');fig.savefig(plot/'Q02_mean_sections.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');ax.plot(m['lat'],mr.mean(axis=0),'o-',label='CTRL');ax.plot(m['lat'],rr.mean(axis=0),'o-',label='GLORYS remapped to same latitude bins');ax.axhline(0,color='gray');ax.set(xlabel='Latitude (N)',ylabel='Mean transport per model latitude cell (Sv)',title='Common wet area: cancellation across latitude');ax.legend();fig.savefig(plot/'Q03_latitude_transport.png',dpi=180);plt.close(fig)
    dump(dest/'summary.json',dict(geometry=geometry,means_Sv={k:float(v.mean()) for k,v in table.items()},metrics=rows,source_signature=sig,filter='Own monthly climatology removed, fourth-order zero-phase Butterworth 2–8 years, no detrending; trim24/36/48',scope='Identical common wet-area comparison; reference full-bottom net unavailable if reference_complete=false. Native model full-bottom provided separately.',script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
