"""Analytical geometry checks, independent source-flux audit and time sampling."""
from compare_channel_transport import *

def main():
    dest=ROOT/'research/transport/2026-11-13_1900m_ctrl'
    # Two bracketing faces with different bottoms, unequal latitude coverage and
    # an independently split reference vertical grid: exact analytical answer.
    m=dict(top=np.zeros((2,1,1)),bottom=np.array([3000.,3500.]).reshape(2,1,1),south=np.array([-.5]),north=np.array([.5]),metric_per_degree=np.array([[100000.],[100000.]]),face_weights=np.array([.5,.5]))
    r=dict(top=np.array([[0.],[2500.]]),bottom=np.array([[2500.],[3200.]]),south=np.array([-.25]),north=np.array([.5]))
    op=common_operator(m,r,1900);expected=.5*(3000-1900)*.75*100000+.5*(3200-1900)*.75*100000
    assert abs(op.sum()-expected)<1e-6
    ma=np.asarray(op.sum(axis=1)).reshape(2,1,1);ra=np.asarray(op.sum(axis=0)).reshape(2,1)
    assert abs(integrate(np.ones((1,2,1,1)),ma)[0][0]-expected/1e6)<1e-10
    u=np.array([2.,-1.]).reshape(1,2,1,1);net,east,west=integrate(u,ma);assert np.isclose(net[0],(.5*2*1100-.5*1300)*.75*100000/1e6);assert east[0]>0 and west[0]<0
    # Model full native section independently integrated from original face fields.
    p=ROOT/'output/261016_ctrl';series=pd.read_csv(dest/'transports.csv',index_col=0,parse_dates=True);dates=series.index;nx,ny,nz=720,192,40
    x=np.fromfile(p/'XG.data',dtype='>f4').reshape(ny,nx)[0].astype(float);yg=np.fromfile(p/'YG.data',dtype='>f4').reshape(ny,nx);dy=np.fromfile(p/'DYG.data',dtype='>f4').reshape(ny,nx);hw=np.fromfile(p/'hFacW.data',dtype='>f4').reshape(nz,ny,nx);rf=-np.fromfile(p/'RF.data',dtype='>f4').astype(float);hi=int(np.searchsorted(x,142));weights=[(x[hi]-142)/(x[hi]-x[hi-1]),(142-x[hi-1])/(x[hi]-x[hi-1])];selected=set(dates[[0,100,335]]);errors=[]
    for path in sorted(p.glob('diag3d.*.meta')):
        nr,fl,t=meta(path)
        if t not in selected:continue
        a=np.memmap(path.with_suffix('.data'),dtype='>f4',mode='r',shape=(nr,nz,ny,nx));value=0.
        for i,w in zip([hi-1,hi],weights):
            for j in range(ny-1):
                south=float(yg[j,i]);north=float(yg[j+1,i]);width=max(0,min(north,2)-max(south,-3.5))
                if width==0:continue
                for k in range(nz):
                    dz=max(0,rf[k]+(rf[k+1]-rf[k])*float(hw[k,j,i])-max(rf[k],1900))
                    value+=w*float(a[fl.index('UVEL'),k,j,i])*dz*float(dy[j,i])*width/(north-south)/1e6
        errors.append(abs(value-series.loc[t,'model_native_net_Sv']))
    assert len(errors)==3 and max(errors)<1e-9,errors
    # Paired temporal resampling; not an independent-observation or model ensemble.
    f=pd.read_csv(dest/'filtered_transports.csv',index_col=0,parse_dates=True);m=f.model_common_net_Sv.values[36:-36];r=f.reference_common_net_Sv.values[36:-36];N=len(m);rng=np.random.default_rng(20261113);rows=[]
    for block in [24,36,48]:
        vals=[]
        for _ in range(2000):
            ix=((rng.integers(N,size=(int(np.ceil(N/block)),1))+np.arange(block))%N).ravel()[:N];vals.append([np.corrcoef(m[ix],r[ix])[0,1],np.std(m[ix])/np.std(r[ix])])
        ci=np.quantile(vals,[.025,.975],axis=0);rows.append(dict(block_months=block,correlation_ci_low=ci[0,0],correlation_ci_high=ci[1,0],std_ratio_ci_low=ci[0,1],std_ratio_ci_high=ci[1,1]))
    pd.DataFrame(rows).to_csv(dest/'paired_block_uncertainty.csv',index=False)
    validation=json.loads((dest/'validation.json').read_text());validation.update(analytical_partial_geometry_area_error_m2=float(abs(op.sum()-expected)),constant_velocity_transport_test=True,opposite_face_signed_flux_test=True,independent_model_source_months=3,independent_model_source_max_error_Sv=max(errors),time_uncertainty='Paired circular blocks 24/36/48 months, 2000 samples; conditional time sampling, not reanalysis uncertainty or an initial-condition ensemble.');dump(dest/'validation.json',validation)
    print('Channel transport validation passed',flush=True);print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
