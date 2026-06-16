import numpy as np
import xarray as xr
import f90nml
from pathlib import Path
from scipy.ndimage import label, distance_transform_edt

# 基本范围和分辨率
xrg = 180
yrg = 48
res = 12
nx, ny = int(xrg*res), int(yrg*res)   # gridpoints in x

print('Nx,Ny=',nx,ny)

x0 = 112     # origin in x,y for ocean domain
y0 = -23.5    # (i.e. southwestern corner of ocean domain)
x1 = x0 + (nx-1)/res    # origin in x,y for ocean domain
y1 = y0 + (ny-1)/res    # (i.e. southwestern corner of ocean domain)
print('x1,y1=',x1,y1)

ipath = Path.home()/'data'
# input files
fbath = ipath / 'GLO-MFC_001_030_mask_bathy.nc'
fcoor = ipath / 'GLO-MFC_001_030_coordinates.nc'
fclim = ipath / 'cmems_climatology_mon.nc'

basedir = Path.home()/'eqp-dlfv-mitgcm-era5'
opath = basedir/'input'

nml = f90nml.read(basedir/'config/data')
nml.repeat_counter=True
nml['PARM03']['deltaT'] = 450.
nml['PARM03']['dumpFreq'] = 0.
nml['PARM04']['delX'] = [1/12]*nx
nml['PARM04']['delY'] = [1/12]*ny
nml['PARM04']['xgOrigin'] = x0-1/12
nml['PARM04']['ygOrigin'] = y0-1/12
# 垂直网格
ds = xr.open_dataset(fcoor)
dr = ds['e3t'].values
Rid = [0,3,6,8,10]+[id for id in range(12,dr.size)]
dR = np.zeros(len(Rid)-1)
for i in range(len(Rid)-1):
    dR[i] = dr[Rid[i]:Rid[i+1]].sum()
nz = len(dR)
print('Nr  = {:4d}'.format(nz))
print('H   =',dR.sum())
print(' delR=', end='')
for i, x in enumerate(dR):
    if i % 10 == 0 and i > 0:
        print()  # New line after every 10 values
        print('      ', end='')  # Add 5 spaces for indentation
    print(f"{x:6.2f},", end='')
print()  # Final newline
nml['PARM04']['delR'] = dR

# 生成地形
MINIMUM_DEPTH = 7.2
with xr.open_dataset(fbath) as ds:
    da = ds['deptho']
    hl = da.loc[y0:y1,x0:x1]
    hr = da.loc[y0:y1,x0-360:x1-360]
    hr['longitude'] = hr['longitude'] + 360
ho = xr.concat([hl,hr],dim='longitude')
h_arr = ho.values
labels, nlabels = label(np.isfinite(h_arr))
sizes = np.bincount(labels.ravel())
sizes[0] = 0
mask = labels != sizes.argmax()
h_arr[mask] = np.nan
h_arr[h_arr < MINIMUM_DEPTH] = MINIMUM_DEPTH
h_arr[np.isnan(h_arr)] = 0
if h_arr.shape != (ny,nx):
    raise ValueError(f"Expected shape {(ny,nx)}, but got {h_arr.shape}")
relpath = 'bathy12.bin'
h_arr.astype('>f4').tofile(opath/relpath)
print(f"Written bathymetry to {relpath}")
nml['PARM05']['bathyFile'] = relpath


# 生成初始场
def fillna(arr):
    filled = arr.copy()
    if np.isnan(arr).any():
        mask = np.isnan(arr)
        # 计算每个NaN位置最近非NaN值的位置索引
        idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
        filled[:] = arr[tuple(idx)]
    return filled
var_names = ['uo','vo','thetao','so','zos']
initnmlname = ['uVelInitFile','vVelInitFile','hydrogThetaFile','hydrogSaltFile','pSurfInitFile']
oinit = 'init/%s_12.bin'
result = {}
with xr.open_dataset(fclim) as ds:
    for iv, var in enumerate(var_names):
        # 12、1月气候态平均作为初始场
        data = ds[var][[0,11]]
        data = data.mean('time')
        data = data.sel(latitude=slice(y0,y1),longitude=slice(x0,x1))
        data = data.values
        if var == 'zos':
            data[mask] = np.nan
            datai = fillna(data)
        else:    
            data[:,mask] = np.nan
            for i in range(data.shape[0]):
                data[i] = fillna(data[i])
            datai = np.zeros((nz,ny,nx))
            for i in range(nz):
                datai[i] = (data[Rid[i]:Rid[i+1]]*dr[Rid[i]:Rid[i+1],None,None]).sum(0)/dR[i,None,None]
        result[var] = datai
        print(f"{var} shape: {datai.shape}")
        relpath = oinit%var
        result[var].astype('>f4').tofile(opath/relpath)
        nml['PARM05'][initnmlname[iv]] = relpath
nml.write(basedir/'config/data.gen',force=True)
# 生成气候态侧边界

obcs = ['S','N','W']
varname = ['thetao','so','uo','vo']
xso_map = {'W': x0,'E': x1}
yso_map = {'S': y0,'N': y1}
mask_obc = {'S':mask[0, :],'N':mask[-1, :],'W':mask[:, 0],'E':mask[:, -1]}
obnml = f90nml.read(basedir/'config/data.obcs')
obnml['OBCS_PARM03']['spongeThickness'] = 25
with xr.open_dataset(fclim) as ds:
    for obc in obcs:
        print(f'处理{obc}边界')
        xso = xso_map.get(obc, slice(x0,x1))
        yso = yso_map.get(obc, slice(y0,y1))
        for vn in varname:
            arr = ds[vn].loc[:,:,yso,xso].values
            arr[:,:,mask_obc[obc]] = np.nan
            # 填充缺失值
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr[i,j] = fillna(arr[i,j])
            # 垂向加权平均（T/S/U/V都处理）
            arr_new = np.zeros((arr.shape[0],nz, arr.shape[2]))
            for k in range(nz):
                arr_new[:,k] = (arr[:,Rid[k]:Rid[k+1]]*dr[Rid[k]:Rid[k+1],None]).sum(1)/dR[k,None]
            relpath = "obcs/OB%s%s_12_%d.bin"%(obc,vn[0],nz)
            obnml['OBCS_PARM01'][f'OB{obc}{vn[0]}File']=relpath
            print(f'{relpath} shape:', arr_new.shape)
            arr_new.astype('>f4').tofile(opath/relpath)
obnml.write(basedir/'config/data.obcs.gen',force=True)
