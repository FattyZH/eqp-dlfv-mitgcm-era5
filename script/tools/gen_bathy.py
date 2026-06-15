from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion, label


# -------------------- user settings --------------------
# Input/output
FBATH = "../../../data/GLO-MFC_001_030_mask_bathy.nc"
OUT_BATHY = "../../input/bathy_avg.bin"

# Model grid
XRG = 180.0
YRG = 48.0
DX = 0.25
DY = 0.25
X0 = 112.0
Y0 = -23.5

# Source-data crop is derived from the model-grid outer edges.
SOURCE_BUFFER_CELLS = 1

# Resampling and bathymetry rules
PRE_KEEP_LARGEST_OCEAN = True
WET_FRAC_THRESHOLD = 0.50
MINIMUM_DEPTH = 7.2
DEPTH_STAT = "mean"  # "mean", "median", "p75", or "max"
DEPTH_FOR_WET_CELLS = "wet_only"  # "wet_only", "area_mean", or "blended".

# Post-coarsening cleanup on the final model grid.
REMOVE_TINY_WET = False
MAKE_PLOT = True


def largest_connected_region(mask):
    labels, nlabels = label(mask)
    if nlabels == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def open_deptho(path, xll, xrr, yd, yu):
    with xr.open_dataset(path) as ds:
        da = ds["deptho"]
        left = da.loc[yd:yu, xll:xrr]
        right = da.loc[yd:yu, xll - 360 : xrr - 360]
        right = right.assign_coords(longitude=right.longitude + 360)
        return xr.concat([left, right], dim="longitude").sortby("longitude").load()


def coarsen_depth_to_model_grid(da, x_centers, y_centers, dx, dy, stat):
    lon = da.longitude.values
    lat = da.latitude.values
    src = da.values.astype(float)

    xlo = x_centers[0] - dx / 2
    xhi = x_centers[-1] + dx / 2
    ylo = y_centers[0] - dy / 2
    yhi = y_centers[-1] + dy / 2
    ii = np.where((lon >= xlo) & (lon < xhi))[0]
    jj = np.where((lat >= ylo) & (lat < yhi))[0]
    src = src[np.ix_(jj, ii)]

    source_dx = float(np.median(np.diff(lon)))
    source_dy = float(np.median(np.diff(lat)))
    block_x = int(round(dx / source_dx))
    block_y = int(round(dy / source_dy))
    ny, nx = y_centers.size, x_centers.size
    expected_shape = (ny * block_y, nx * block_x)
    if src.shape != expected_shape:
        raise ValueError(f"source block shape {src.shape} != expected {expected_shape}")

    blocks = src.reshape(ny, block_y, nx, block_x).transpose(0, 2, 1, 3)
    wet_mask = np.isfinite(blocks)
    wet_count = wet_mask.sum(axis=(2, 3))
    cell_count = block_y * block_x
    wet_frac = wet_count / cell_count

    # Land as zero gives the true area-mean depth over the target cell.
    depth_area_mean = np.nan_to_num(blocks, nan=0.0).mean(axis=(2, 3))

    if stat == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            depth_wet_only = depth_area_mean / wet_frac
    elif stat == "median":
        depth_wet_only = np.nanmedian(blocks, axis=(2, 3))
    elif stat == "p75":
        depth_wet_only = np.nanpercentile(blocks, 75, axis=(2, 3))
    else:
        raise ValueError(f"unknown depth statistic: {stat}")

    depth_wet_only = np.where(wet_count > 0, depth_wet_only, np.nan)
    return depth_wet_only, depth_area_mean, wet_frac


# -------------------- main workflow --------------------
nx, ny = int(XRG // DX), int(YRG // DY)
x1 = X0 + (nx - 1) * DX
y1 = Y0 + (ny - 1) * DY
x_centers = X0 + np.arange(nx) * DX
y_centers = Y0 + np.arange(ny) * DY

source_buffer_x = SOURCE_BUFFER_CELLS * DX
source_buffer_y = SOURCE_BUFFER_CELLS * DY
xll = x_centers[0] - DX / 2 - source_buffer_x
xrr = x_centers[-1] + DX / 2 + source_buffer_x
yd = y_centers[0] - DY / 2 - source_buffer_y
yu = y_centers[-1] + DY / 2 + source_buffer_y

ho = open_deptho(FBATH, xll=xll, xrr=xrr, yd=yd, yu=yu)

# Remove source-grid wet regions that are already disconnected before coarsening.
# This protects narrow high-resolution land barriers from disappearing during resampling.
source_wet = np.isfinite(ho.values)
if PRE_KEEP_LARGEST_OCEAN:
    source_main_ocean = largest_connected_region(source_wet)
    source_wet_removed = source_wet & ~source_main_ocean
    ho = ho.where(source_main_ocean)
else:
    source_wet_removed = np.zeros_like(source_wet, dtype=bool)

# Separate land/sea decision from depth statistics.
depth_wet_only, depth_area_mean, wet_frac = coarsen_depth_to_model_grid(
    ho,
    x_centers=x_centers,
    y_centers=y_centers,
    dx=DX,
    dy=DY,
    stat=DEPTH_STAT,
)

if DEPTH_FOR_WET_CELLS == "wet_only":
    depth = depth_wet_only.copy()
elif DEPTH_FOR_WET_CELLS == "area_mean":
    depth = depth_area_mean.copy()
else:
    raise ValueError(f"unknown DEPTH_FOR_WET_CELLS: {DEPTH_FOR_WET_CELLS}")

wet = wet_frac >= WET_FRAC_THRESHOLD

wet = largest_connected_region(wet)

depth = np.where(wet, depth, np.nan)

# Final cleanup is done on the actual model grid.


shallow_keep = wet & (depth < MINIMUM_DEPTH)
depth[shallow_keep] = MINIMUM_DEPTH

if REMOVE_TINY_WET:
    wet_opened = binary_dilation(binary_erosion(wet))
    tiny_wet = wet & ~wet_opened
    depth[tiny_wet] = 0.0
    wet[tiny_wet] = False
else:
    tiny_wet = np.zeros_like(wet)

bathy = -depth.astype(float)
bathy[np.isnan(depth)] = 0.0

Path(OUT_BATHY).parent.mkdir(parents=True, exist_ok=True)
bathy.astype(dtype=">f4").tofile(OUT_BATHY)

print("source shape:", ho.shape)
print("target shape:", bathy.shape, "expected:", (ny, nx))
print("target lon range:", X0, x1)
print("target lat range:", Y0, y1)
print("source crop:", xll, xrr, yd, yu)
print("removed source wet cells:", int(source_wet_removed.sum()))
print("wet fraction range:", float(np.nanmin(wet_frac)), float(np.nanmax(wet_frac)))
print("wet cells:", int(wet.sum()), "/", wet.size)
print("depth for wet cells:", DEPTH_FOR_WET_CELLS)
print("deepened shallow cells:", int(shallow_keep.sum()))
print("removed tiny wet cells:", int(tiny_wet.sum()))
if wet.any():
    print("min/max wet depth:", float(depth[wet].min()), float(depth[wet].max()))
print("wrote:", OUT_BATHY)

if MAKE_PLOT:
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    im0 = axs[0].pcolormesh(x_centers, y_centers, wet_frac, shading="auto", vmin=0, vmax=1)
    axs[0].set_title("wet fraction")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].pcolormesh(x_centers, y_centers, depth, shading="auto")
    axs[1].set_title("positive depth")
    fig.colorbar(im1, ax=axs[1])
    plt.savefig('bathy_avg.png', dpi=150)
