#%%
import sys
sys.path.append ('../')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio
import xarray as xr
latlim = slice(-5.5,5.5)
import mit_utils

exp = '260410_164626'
hbl = mit_utils.open_mds('../../output/'+exp,prefix='KPPhbl').sel(YC=slice(-2,2))
viscAz = mit_utils.open_mds('../../output/'+exp,prefix='KPPviscAz').sel(YC=slice(-2,2))
display(viscAz)
#%%
ind = slice(-130,-10)
plt.figure(figsize=(10,8))
plt.subplot(211)
hblm = hbl['KPPhbl'][ind].mean('YC')
hblm.mean('time').plot()
plt.subplot(212)
viscAzm = viscAz['KPPviscAz'][ind,:].mean('YC')
viscAzm.mean('time').plot(cmap='RdYlBu_r')
#%%
plt.contourf(hblm.values.reshape(10,12,-1).mean(0))
plt.colorbar()
#%%
import matplotlib.colors as colors

kzs = viscAzm.values.reshape(-1,12,viscAzm.shape[1],viscAzm.shape[2]).mean(0)
for i in range(0,12,3):
    plt.subplot(2,2,i+1)
    plt.contourf(np.log10(kzs[3,:,:]), levels=np.linspace(-5.5, -1.5, 9),extend='both', cmap='RdYlBu_r')
    plt.gca().invert_yaxis()
    plt.colorbar()
