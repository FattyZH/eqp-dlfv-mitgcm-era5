import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

path = '/mnt/d/project/IAVNNG/Data/'
file = 'cmems_climatology_mon.nc'
fn = path + file
ds = xr.open_dataset(fn)
print(ds)
