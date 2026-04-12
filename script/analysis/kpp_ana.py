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
from scipy.interpolate import interp1d
hbls = hblm.values.reshape(10,12,-1).mean(0)
plt.contourf(hbls)
plt.colorbar()
z = viscAz['Zl'].values
x = viscAz['XC'].values
zp1 = viscAz['Zp1'].values
hbn = interp1d(-zp1,np.arange(zp1.size))(hbls)
#%%
import matplotlib.colors as colors

kzs = viscAzm.values.reshape(-1,12,viscAzm.shape[1],viscAzm.shape[2]).mean(0)
cnt = 0
plt.figure(figsize=(10,14))
for i in range(0,12,3):
    cnt+=1
    plt.subplot(4,1,cnt)
    plt.contourf(x,-z,np.log10(kzs[i,:,:]), levels=np.linspace(-5.5, -1.5, 4*4+1),extend='both', cmap='RdYlBu_r')
    plt.plot(x,hbls[i])
    plt.gca().invert_yaxis()
    plt.colorbar()

#%%
i,ix = 3,400
plt.plot(np.log10(kzs[i,:,ix]),-z,marker='.')
plt.axhline(hbls[i,ix],color='k')
plt.axhline(115,color='k')
plt.xlim(-6,-1.5)
plt.ylim(500,0)
