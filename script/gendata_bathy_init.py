from MITgcmutils.utils import writebin
import numpy as np
from numpy import cos, pi
import xarray as xr

Ho = 4000  # depth of ocean (m)
nx = 170    # gridpoints in x
ny = 30    # gridpoints in y
dx = 1     # grid spacing in x (degrees longitude)
dy = 1     # grid spacing in y (degrees latitude)
x0 = 120     # origin in x,y for ocean domain
y0 = -15    # (i.e. southwestern corner of ocean domain)
x1 = x0 + (nx-1)*dx    # origin in x,y for ocean domain
y1 = y0 + (ny-1)*dy    # (i.e. southwestern corner of ocean domain)
opath = '../input/'
ipath = '/mnt/d/project/IAVNNG/Data/'
fbath = 'GLO-MFC_001_030_mask_bathy.nc'
fclim = 'cmems_climatology_mon.nc'
inc = 12

ds = xr.open_dataset(ipath+fbath)
h = ds['deptho'].loc[y0:y1:inc,x0:x1:inc]
h = h.values
writebin(opath+'bath.bin',h)

# 

