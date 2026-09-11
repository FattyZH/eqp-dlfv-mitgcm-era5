from mitkit.paths import project_root
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from mitkit.io import open_mds

keep_vars = ['UVEL', 'VVEL', 'THETA', 'SALT']
expname = '260821_ctrl'

inpath = project_root() / 'output'
outpath = project_root() / 'input/rbcs'
exp = inpath/expname
ds = open_mds(exp,prefix='diag3d').sel(time=slice('1996','2025'))
ds = ds[keep_vars]

# 数据第一个时间点对应的月份
first_month = int(ds.time.dt.month.isel(time=0))

clim = xr.concat(
    [
        ds.isel(
            time=slice((month - first_month) % 12, None, 12)
        ).mean("time", skipna=True)
        for month in range(1, 13)
    ],
    dim=xr.IndexVariable("month", np.arange(1, 13))
)
for i in range(len(keep_vars)):
    clim[keep_vars[i]].astype(">f4").values.tofile(outpath/f"rbcs_{keep_vars[i][0]}_clim.bin")
