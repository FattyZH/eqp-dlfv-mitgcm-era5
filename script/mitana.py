import MITgcmutils as mutils
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
path = '../output/251108/'
a = xr.open_dataset(path + 'dync.nc')
print(a)
# id = a.indexes['T'].get_loc('1940-03-01 00:00:00')
id = -1
plt.figure(figsize=(18,14))
plt.subplot(4,1,1)
a['UVEL'][id,0].plot(cmap='RdBu_r',vmin=-1.5, vmax=1.5)
plt.subplot(4,1,2)
a['VVEL'][id,0].plot(cmap='RdBu_r',vmin=-1.5, vmax=1.5)
plt.subplot(4,1,3)
a['THETA'][id,0].plot(cmap='RdBu_r',vmin=15, vmax=35)
plt.subplot(4,1,4)
a['SALT'][id,0].plot(cmap='RdBu_r',vmin=30, vmax=38)
plt.savefig(path+'dync.png',dpi=300)