import xarray as xr
import numpy as np
import os
path = '/mnt/d/project/IAVNNG/Data/era5_tau_mon.nc'
opath = './'
data = xr.open_dataset(path)
x0,x1 = 120,290  # 经度范围
y0,y1 = -15,15  # 纬度范围
inc = 4
# 读取u和v风应力分量
nt = data.dims['valid_time']
tauu = data['ewss'].loc[:,y0:y1:-1*inc,x0:x1:inc] 
tauv = data['nsss'].loc[:,y0:y1:-1*inc,x0:x1:inc]
tauu = tauu[:,:-1,:-1]  # 去掉最后一行和最后一列
tauv = tauv[:,:-1,:-1]  # 去掉最后一行和最后一列

tauu = tauu.values/86400  # era5提供的风应力是一天的积分，需要除以86400转换为风应力
tauv = tauv.values/86400  # 同上
print(tauu.shape, tauv.shape)
# # 保存为MITgcm驱动所需的二进制文件
tauu.astype('>f4').tofile(opath+'taux.bin')
tauv.astype('>f4').tofile(opath+'tauy.bin')