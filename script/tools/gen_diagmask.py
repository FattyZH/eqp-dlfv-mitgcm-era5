import f90nml
import numpy as np
from pathlib import Path
base = Path('../../input')
nml = f90nml.read(base/'data')
grid = nml['parm04']
dx,dy= np.array(grid['delx']),np.array(grid['dely'])
x0,y0= grid['xgorigin'],grid['ygorigin']
x = x0 + np.cumsum(dx) - dx/2
y = y0 + np.cumsum(dy) - dy/2
mask = np.zeros((y.size,x.size))
# select region
ix = np.where(x >= 142)[0][0]
iy = (y >= -2) & (y <= 2)

mask[iy, ix] = 1

mask.astype('>f4').tofile(base/"diag_regmask.bin")