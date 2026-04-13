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

exp = '260412_090002'
# hbl = mit_utils.open_mds('../../output/'+exp,prefix='PPdiffKr').sel(YC=slice(-2,2))
viscAz = mit_utils.open_mds('../../output/'+exp,prefix='PPviscAr').sel(YC=slice(-2,2))
display(viscAz)
#%%
ind = slice(-130,-10)
viscAzm = viscAz['PPviscAr'][ind,:].mean('YC')
viscAzm.mean('time').plot(cmap='RdYlBu_r')
#%%
import matplotlib.colors as colors
z = viscAz['Zl'].values
x = viscAz['XC'].values
zp1 = viscAz['Zp1'].values
kzs = viscAzm.values.reshape(-1,12,viscAzm.shape[1],viscAzm.shape[2]).mean(0)
cnt = 0
plt.figure(figsize=(10,14))
for i in range(0,12,3):
    cnt+=1
    plt.subplot(4,1,cnt)
    plt.contourf(x,-z,np.log10(kzs[i,:,:]), levels=1.5+np.linspace(-5.5, -1.5, 4*4+1),extend='both', cmap='RdYlBu_r')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.ylim(400,0)

#%%
i,ix = 6,300
plt.plot(np.log10(kzs[i,:,ix]),-z,marker='.')
plt.axhline(115,color='k')
plt.xlim(-6,-1.5)
plt.ylim(500,0)
