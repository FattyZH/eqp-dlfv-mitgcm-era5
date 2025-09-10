import xarray as xr
import numpy as np
import os
opath = './'
nx = 170
ny = 30
nn = 13
tauu = np.zeros((nn,ny,nx))
tauu = tauu+np.arange(0,nx)[None,None,:]/100
tauv = np.zeros((nn,ny,nx))
tauv = tauv+np.arange(0,ny)[None,:,None]/100

print(tauu.shape)

tauu.astype('>f4').tofile(opath+'taux.bin')
tauv.astype('>f4').tofile(opath+'tauy.bin')