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

exp = '260409_162411'
hbl = mit_utils.open_mds('../../output/'+exp,prefix='KPPhbl')
viscAz = mit_utils.open_mds('../../output/'+exp,prefix='KPPviscAz')
#%%
display(viscAz)
ind = 9+12*16
plt.figure(figsize=(10,8))
plt.subplot(211)
hbl['KPPhbl'][ind].plot(cmap='RdYlBu_r',vmax=120)
plt.subplot(212)
viscAz['KPPviscAz'][ind,1].plot(cmap='RdYlBu_r',vmax=0.03)
