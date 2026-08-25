from MITgcmutils.utils import writebin
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import os
work_dir = Path(os.environ['WORK_DIR'])
opath = work_dir / 'input/exf/era5_mclim/'

ipath = Path('~/data/')
fsflux = ipath / 'era5_surfflux_mon.nc'
fsvar = ipath / 'era5_surfvar_mon.nc'

do_clim = True # 是否气候态

# 时间和空间切片
tsl = slice('1992-01', '2024-12') # 时间切片
ylim = slice(-30,30,-1)
xlim = slice(105,295)

ds = xr.open_dataset(fsvar).sel(valid_time=tsl,latitude=ylim,longitude=xlim)
print(ds)
var_names = ['u10','v10','t2m','sh2m','avg_tprate','avg_sdswrf','avg_sdlwrf']
# 存储处理后数据的字典
result = {}
for var in var_names:
    data = ds[var].load()
    # 如果需要气候态（月平均）
    if do_clim:
        data = data.groupby('valid_time.month').mean('valid_time')
    # 转为 numpy 数组并存入字典
    result[var] = data.values
    print(f"{var} shape: {result[var].shape}")

wspeed = np.sqrt(result['u10']**2 + result['v10']**2) # 风速（用于计算感热、潜热和蒸发）
aqh = result['sh2m']  # 比湿
swdown = result['avg_sdswrf']  # 单独的短波辐射通量
lwdown = result['avg_sdlwrf']  # 下行长波辐射
precip = result['avg_tprate']/1e3 # 单位转换,mit以淡水流失为正（era5的以淡水输入为正）

# 保存为MITgcm驱动所需的二进制文件
writebin(opath/'uwind.bin',result['u10'])
writebin(opath/'vwind.bin',result['v10'])
writebin(opath/'wspeed.bin',wspeed)
writebin(opath/'atemp.bin',result['t2m'])
writebin(opath/'aqh.bin', aqh)
writebin(opath/'swdown.bin',swdown)
writebin(opath/'lwdown.bin',lwdown)
writebin(opath/'precip.bin',precip)
