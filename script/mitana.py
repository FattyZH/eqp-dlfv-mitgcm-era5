import MITgcmutils as mutils
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mit_utils import open_mds
from pathlib import Path

outdir = '../output/'
dirs = [p for p in Path(outdir).iterdir() if p.is_dir()]
dirs.sort()
path = dirs[-1]

a = open_mds(path,prefix=['U','V','T','S'],ref_date='1940-01-16')
print(a)
# id = a.indexes['T'].get_loc('1940-03-01 00:00:00')

id = -1
lev = 0
plt.figure(figsize=(12,12))
plt.subplot(4,1,1)
a['U'][id,lev].plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,2)
a['V'][id,lev].plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,3)
a['T'][id,lev].plot(cmap='RdBu_r',vmin=15, vmax=35)
plt.subplot(4,1,4)
a['S'][id,lev].plot(cmap='RdBu_r',vmin=30, vmax=38)
plt.savefig(Path.joinpath(path,'_state_last.png'),dpi=300)


id = slice(-12,None)
lev = 0
plt.figure(figsize=(12,12))
plt.subplot(4,1,1)
a['U'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,2)
a['V'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=-1, vmax=1)
plt.subplot(4,1,3)
a['T'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=15, vmax=35)
plt.subplot(4,1,4)
a['S'][id,lev].mean('time').plot(cmap='RdBu_r',vmin=30, vmax=38)
plt.savefig(Path.joinpath(path,'_state_mean.png'),dpi=300)
