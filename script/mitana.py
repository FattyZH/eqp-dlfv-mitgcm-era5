import MITgcmutils as mutils
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
path = '../output/251111/'
a = xr.open_dataset(path + 'state.nc')
print(a)
# id = a.indexes['T'].get_loc('1940-03-01 00:00:00')
id = -1
lev = 0
plt.figure(figsize=(18,14))
plt.subplot(4,1,1)
a['U'][id,lev].plot(cmap='RdBu_r',vmin=-1.5, vmax=1.5)
plt.subplot(4,1,2)
a['V'][id,lev].plot(cmap='RdBu_r',vmin=-1.5, vmax=1.5)
plt.subplot(4,1,3)
a['Temp'][id,lev].plot(cmap='RdBu_r',vmin=15, vmax=35)
plt.subplot(4,1,4)
a['S'][id,lev].plot(cmap='RdBu_r',vmin=30, vmax=38)
plt.savefig(path+'state.png',dpi=300)