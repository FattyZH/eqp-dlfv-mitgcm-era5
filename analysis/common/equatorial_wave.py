"""Conditional long-wave fitting helpers; coefficients are not reflection efficiencies."""
from analysis_common import *
from scipy.special import eval_hermite, factorial
DEST=RETAINED/'research/mechanism/2026-10-20_eastern_absorber_audit'
CONFIGS=[('y7p5_R3',7.5,3),('y5_R3',5.,3),('y7p5_R5',7.5,5),('y5_R5',5.,5)]
PAIRS={'C':EXPS['C'],'A':EXPS['A']}
BETA=2*7.292115e-5/6371000.
RADIUS=6371000.

def hermite(j,y):return eval_hermite(j,y)*np.exp(-y*y/2)/np.sqrt(2.**j*factorial(j)*np.sqrt(np.pi))

def wave_basis(lat,c,nr):
 eta=RADIUS*np.deg2rad(lat)/np.sqrt(c/BETA);q0=hermite(0,eta);u=[q0/np.sqrt(2)];p=[q0/np.sqrt(2)]
 for j in range(1,nr+1):
  scale=np.sqrt(2*(j+1)/(2*j+1));q=scale*np.sqrt(j/(j+1))*hermite(j+1,eta);r=scale*hermite(j-1,eta);u.append((q-r)/2);p.append((q+r)/2)
 return np.concatenate([np.stack(u,axis=1),np.stack(p,axis=1)],axis=0)

def prepare(g,b):
 # Read enough of U on adjacent faces, fit on C coordinates.
 ii=np.flatnonzero((g['xc']>=220)&(g['xc']<=266));js=np.flatnonzero(abs(g['y'])<=7.5);islice=slice(ii[0],ii[-1]+1);islice_u=slice(ii[0],ii[-1]+2);jslice=slice(js[0],js[-1]+1);nj=len(js);ni=len(ii);nm=5;nb=6;nz=len(b.depth)
 phi=b.velocity_mode.values[:,1:6];quad=b.quadrature_weight.values;projector=(phi*quad[:,None]).T/quad.sum();c=b.phase_speed.values[1:6];lat=g['y'][js];wet=(g['hc'][:nz,jslice,islice]>0).all(axis=0)&(g['hw'][:nz,jslice,islice]>0).all(axis=0)&(g['hw'][:nz,jslice,slice(ii[0]+1,ii[-1]+2)]>0).all(axis=0)
 operators=np.zeros((len(CONFIGS),nm,ni,nb,2*nj));condition=np.full((len(CONFIGS),nm,ni),np.nan);designs={};samplevalid={}
 for ci,(name,width,nr) in enumerate(CONFIGS):
  for n,speed in enumerate(c):
   B=wave_basis(lat,speed,nr);designs[(ci,n)]=B
   for ix in range(ni):
    keep=wet[:,ix]&(abs(lat)<=width);valid=np.r_[keep,keep];samplevalid[(ci,ix)]=valid
    if keep.sum()<max(12,2*(nr+1)):continue
    sw=np.sqrt(np.r_[np.cos(np.deg2rad(lat)),np.cos(np.deg2rad(lat))][valid]);Bw=B[valid]*sw[:,None];cond=np.linalg.cond(Bw);condition[ci,n,ix]=cond
    if cond>1e5:continue
    op=np.linalg.pinv(Bw)*sw[None];operators[ci,n,ix,:nr+1,valid]=op.T # advanced boolean indexing has (valid,basis) layout
 dump(DEST/'fit_setup.json',{'configs':CONFIGS,'lon_min':g['xc'][ii[0]],'lon_max':g['xc'][ii[-1]],'vertical_modes':[1,2,3,4,5],'vertical_reference':'Frozen 0–3001m basis; provisional for long-wave identification','combined_u_pressure_fit':True,'c_m_s':c.tolist(),'median_condition_number':float(np.nanmedian(condition))})
 return locals()

def extract(key,g,b,setup,masks):
 p=ROOT/'output'/PAIRS[key];dest=DEST/(key+'_wave_components.npz');sig=signature(p);tag=DEST/(key+'_sources.json')
 if dest.exists() and tag.exists() and json.loads(tag.read_text())['signature']==sig:print('reuse',key,flush=True);return dict(np.load(dest))
 ii=setup['ii'];js=setup['js'];ni=len(ii);nj=len(js);nz=len(b.depth);nm=5;projector=setup['projector'];operators=setup['operators'];c=setup['c'];gates=np.array([230,240,250,254,258,262]);gateidx=np.array([np.argmin(abs(g['xc'][ii]-v)) for v in gates]);actualgates=g['xc'][ii[gateidx]]
 point_x=np.array([np.argmin(abs(g['x']-v)) for v in [253,257,262]]);point_k=np.array([np.argmin(abs(g['z']-v)) for v in [500,1000,2000]]);jeq=np.argmin(abs(g['y']));jeqlocal=int(np.where(js==jeq)[0][0]);k0,k1=point_k.min(),point_k.max()+1
 def read(m):
  nr,fields,t=meta(m);a=np.memmap(m.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,NZ,NY,NX));u=np.array(a[fields.index('UVEL'),:nz,setup['jslice'],setup['islice_u']],float);points=u[point_k,jeqlocal][:,point_x-ii[0]];u=(u[:,:,:-1]+u[:,:,1:])/2
  ph=np.array(a[fields.index('PHIHYD'),:nz,setup['jslice'],setup['islice']],float);del a
  um=(projector@u.reshape(nz,-1)).reshape(nm,nj,ni);pm=(projector@ph.reshape(nz,-1)).reshape(nm,nj,ni)/c[:,None,None];state=np.concatenate([um,pm],axis=1)
  coeff=np.einsum('cnxby,nyx->cnbx',operators,state,optimize=True)
  f=m.with_name(m.name.replace('diag3d','diag_mom_u'));nr,fl,dt=meta(f);assert dt==t;a=np.memmap(f.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,NZ,NY,NX));v=np.array(a[fl.index('Um_Ext'),k0:k1,jeq,:],float);ext=v[point_k-k0][:,point_x];del a
  return t.to_datetime64(),coeff.astype(np.float32),state[:,:,gateidx].astype(np.float32),points,ext
 results=[]
 with ThreadPoolExecutor(max_workers=int(os.environ.get('EQP_IO_WORKERS','16'))) as pool:
  for i,r in enumerate(pool.map(read,sorted(p.glob('diag3d.*.meta')))):
   results.append(r)
   if (i+1)%60==0:print('wave extract',key,i+1,'/420',flush=True)
 times,coef,snap,Upt,Extpt=map(np.stack,zip(*results));out=dict(time=times,coef=coef,state_gates=snap,u_points=Upt,ext_points=Extpt,gate_lons=actualgates,lon=g['xc'][ii],lat=g['y'][js],point_lons=g['x'][point_x],point_depths=g['z'][point_k],point_mask=masks['U'][point_k,jeq][:,point_x]);np.savez_compressed(dest,**out);dump(tag,{'signature':sig,'algorithm':'conditional_longwave_v1'});return out
