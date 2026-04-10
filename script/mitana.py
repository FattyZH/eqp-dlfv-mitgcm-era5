import MITgcmutils as mutils
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mit_utils import open_mds
from pathlib import Path

outdir = 'output/'
dirs = [p for p in Path(outdir).iterdir() if p.is_dir()]
dirs.sort()
path = dirs[-4]
name = path.name
a = open_mds(path,prefix=['dync'])
print(a)
# id = a.indexes['T'].get_loc('1940-03-01 00:00:00')

id = -1
lev = 0
plt.figure(figsize=(12,12))
plt.subplot(4,1,1)
a['UVEL'][id,lev].plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,2)
a['VVEL'][id,lev].plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,3)
a['THETA'][id,lev].plot(cmap='RdBu_r',vmin=15, vmax=35)
plt.subplot(4,1,4)
a['SALT'][id,lev].plot(cmap='RdBu_r',vmin=30, vmax=38)
plt.savefig(outdir+name+'_state_last.png',dpi=300)


id = slice(-12,None)
lev = 0
plt.figure(figsize=(12,12))
plt.subplot(4,1,1)
a['UVEL'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,2)
a['VVEL'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,3)
a['THETA'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=15, vmax=35)
plt.subplot(4,1,4)
a['SALT'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=30, vmax=38)
plt.savefig(outdir+name+'_state_mean.png',dpi=300)
