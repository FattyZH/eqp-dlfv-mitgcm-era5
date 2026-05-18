import subprocess
import time
from contextlib import contextmanager
import xarray as xr
import fsspec
import numpy as np

@contextmanager
def rclone_serve(remote, port=8080):
    proc = subprocess.Popen(
        ["rclone", "serve", "http", remote, "--addr", f"127.0.0.1:{port}"],
    )
    time.sleep(2)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait()

RPATH = "JRA55-do"
REMOTE = "nas054:data"

fvars = {
    "uwind": "uas",
    "vwind": "vas",
    "atemp": "tas",
    "aqh":"huss",
    "swdown":"rsds",
    "lwdown":"rlds",
    "precip":"prra",
}

f_type = {
    "uas":  0,
    "vas":  0,
    "tas":  0,
    "huss": 0,
    "rsds": 1,
    "rlds": 1,
    "prra": 1,
}

def make_fname(var, yr, vtype):
    t_start = "0130" if vtype else "0000"
    t_end   = "2230" if vtype else "2100"
    return (
        f"{var}_input4MIPs_atmosphericState_OMIP"
        f"_MRI-JRA55-do-1-6-0_gr"
        f"_{yr}0101{t_start}-{yr}1231{t_end}.nc"
    )

# 区域参数
LON_W, LON_E = 108.0, 297.0
LAT_S, LAT_N = -30.0,  30.0

OUTPUT_DIR = "../../data/exf/JRA55"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_region(da):
    return da.sel(
        lon=slice(LON_W, LON_E),
        lat=slice(LAT_S, LAT_N),
    )

def process_var(var, yr, base_url, print_grid=False):
    fname = make_fname(fvars[var], yr, f_type[fvars[var]])
    url   = f"{base_url}/{RPATH}/{fname}"

    with fsspec.open(url, "rb") as f:
        ds = xr.open_dataset(f)
        da = extract_region(ds[fvars[var]])
        arr = da.values
        if print_grid:
            lons = da.lon.values
            lats = da.lat.values

    arr = da.values  # (time, lat, lon)，time = 365*8
    arr = arr.reshape(-1, 8, arr.shape[1], arr.shape[2]).mean(axis=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_fname = os.path.join(OUTPUT_DIR, f"{var}_{yr}")
    arr.astype(">f4").tofile(out_fname)
    print(f"[OK] {out_fname}  shape={arr.shape}")
    if print_grid:
        lon_inc = np.diff(lons)
        lat_inc = np.diff(lats)
        lon_uniform = np.allclose(lon_inc, lon_inc[0])
        lat_uniform = np.allclose(lat_inc, lat_inc[0])
        if not lon_uniform:
            print(f"  {var}_lon_inc = 不均匀! min={lon_inc.min():.4f} max={lon_inc.max():.4f}")
        if not lat_uniform:
            print(f"  {var}_lat_inc = 不均匀! min={lat_inc.min():.4f} max={lat_inc.max():.4f}")
        lines = []
        lines.append(f" {var}_lon0    = {lons[0]:.4f}")
        lines.append(f" {var}_lon_inc = {(lons[-1] - lons[0]) / (len(lons) - 1):.4f}")
        lines.append(f" {var}_lat0    = {lats[0]:.4f}")
        lines.append(f" {var}_lat_inc = {(lats[-1] - lats[0]) / (len(lats) - 1):.4f}")
        lines.append(f" {var}_nlon    = {len(lons)}")
        lines.append(f" {var}_nlat    = {len(lats)}")
        grid_info_path = os.path.join(OUTPUT_DIR, "grid_info.txt")
        with open(grid_info_path, "a+") as gf:
            gf.write("\n".join(lines) + "\n")
        print(f"[INFO] 网格信息已写入 {grid_info_path}")

with rclone_serve(REMOTE) as base_url:
    for i, yr in enumerate(range(1990, 2025)):
        for var in fvars.keys():
            process_var(var, yr, base_url, print_grid=(i == 0))