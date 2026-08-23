from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import label
import os

# ============================================================
# User settings
# ============================================================
FBATH = Path("~/data/GLO-MFC_001_030_mask_bathy.nc")
WORK_DIR = Path(os.environ["WORK_DIR"])
OUT_BATHY = WORK_DIR / "input/bathy_avg.bin"
OUT_FIGURE = Path("bathy_avg.png")

# MITgcm horizontal grid
XRG = 180.0
YRG = 48.0
DX = 0.25
DY = 0.25
X0 = 112.0
Y0 = -23.5

# A model cell is wet when at least this fraction of source cells is wet.
WET_FRAC_THRESHOLD = 0.50

# Minimum positive water depth written to MITgcm.
MINIMUM_DEPTH = 5

# Small tolerance used only to avoid coordinate-selection roundoff.
COORD_TOL = 1.0e-6

MAKE_PLOT = True

# MITgcm horizontal cells exchange through cell faces, not corners.
FOUR_NEIGHBOUR_STRUCTURE = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)


# ============================================================
# Connectivity
# ============================================================
def largest_connected_region(mask):
    """
    Retain the largest four-neighbour connected True region.
    """
    mask = np.asarray(mask, dtype=bool)

    labels, nlabels = label(
        mask,
        structure=FOUR_NEIGHBOUR_STRUCTURE,
    )

    if nlabels == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    return labels == sizes.argmax()


def largest_region_ignoring_outer_ring(mask):
    """
    Determine the principal connected ocean without allowing the outermost
    model-grid ring to connect otherwise disconnected water bodies.

    The outermost ring is temporarily treated as land. After identifying
    the largest connected interior ocean, boundary cells are restored only
    when they connect directly inward to that ocean.

    This is useful when a water body is connected to the principal ocean
    only through one layer of cells along an OBCS/sponge boundary.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")

    if min(mask.shape) < 3:
        raise ValueError("mask is too small to remove the outer ring")

    interior_mask = mask.copy()

    # Temporarily remove the complete outer ring.
    interior_mask[0, :] = False
    interior_mask[-1, :] = False
    interior_mask[:, 0] = False
    interior_mask[:, -1] = False

    keep = largest_connected_region(interior_mask)

    # Restore boundary cells only when the immediately inward cell belongs
    # to the principal interior ocean.
    keep[0, 1:-1] = mask[0, 1:-1] & keep[1, 1:-1]
    keep[-1, 1:-1] = mask[-1, 1:-1] & keep[-2, 1:-1]

    keep[1:-1, 0] = mask[1:-1, 0] & keep[1:-1, 1]
    keep[1:-1, -1] = mask[1:-1, -1] & keep[1:-1, -2]

    # Restore corners only when connected to an already restored side cell.
    keep[0, 0] = mask[0, 0] & (keep[0, 1] | keep[1, 0])
    keep[0, -1] = mask[0, -1] & (keep[0, -2] | keep[1, -1])
    keep[-1, 0] = mask[-1, 0] & (keep[-1, 1] | keep[-2, 0])
    keep[-1, -1] = mask[-1, -1] & (
        keep[-1, -2] | keep[-2, -1]
    )

    return keep


# ============================================================
# Source bathymetry
# ============================================================
def open_deptho(path, xll, xrr, yd, yu):
    """
    Open and crop source bathymetry to the actual model-domain boundaries.

    The longitude coordinate is converted to [0, 360), then duplicated at
    longitude +/-360 so that continuous longitude ranges can be selected
    even when a model domain crosses the 0/360 seam.
    """
    with xr.open_dataset(path) as ds:
        if "deptho" not in ds:
            raise KeyError(f"'deptho' was not found in {path}")

        da = ds["deptho"]

        required_coords = {"longitude", "latitude"}
        missing_coords = required_coords - set(da.coords)

        if missing_coords:
            raise KeyError(
                f"missing coordinates: {sorted(missing_coords)}"
            )

        # Ensure ascending latitude.
        da = da.sortby("latitude")

        # Standardise longitude to [0, 360).
        longitude = np.mod(
            da.longitude.values.astype(float),
            360.0,
        )
        da = da.assign_coords(longitude=longitude)
        da = da.sortby("longitude")

        # Remove a possible duplicate longitude such as both 0 and 360.
        rounded_lon = np.round(
            da.longitude.values,
            decimals=10,
        )
        _, unique_indices = np.unique(
            rounded_lon,
            return_index=True,
        )
        da = da.isel(longitude=np.sort(unique_indices))

        # Extend longitude coordinates for seam-safe selection.
        da_extended = xr.concat(
            [
                da.assign_coords(longitude=da.longitude - 360.0),
                da,
                da.assign_coords(longitude=da.longitude + 360.0),
            ],
            dim="longitude",
        ).sortby("longitude")

        da_crop = da_extended.sel(
            longitude=slice(xll, xrr),
            latitude=slice(yd, yu),
        )

        if da_crop.longitude.size == 0:
            raise ValueError(
                "source longitude selection is empty; "
                "check the model longitude range"
            )

        if da_crop.latitude.size == 0:
            raise ValueError(
                "source latitude selection is empty; "
                "check the model latitude range"
            )

        return da_crop.load()


# ============================================================
# Grid coarsening
# ============================================================
def coarsen_depth_to_model_grid(
    da,
    x_centers,
    y_centers,
    dx,
    dy,
):
    """
    Coarsen regular source-grid bathymetry to the MITgcm model grid.

    The land/sea decision and water-depth calculation are handled
    separately.

    Wet fraction
    ------------
    Fraction of source-grid cells containing ocean within each model cell.

    Final depth
    -----------
    The depth is the geometric mean between:

    1. Mean depth over wet source cells.
    2. Area-equivalent depth, with source land treated as zero.

    Therefore:

        depth = depth_wet_only * sqrt(wet_fraction)

    This avoids giving a partially wet model cell the full wet-only depth,
    while avoiding the excessive shoaling produced by a pure area mean.
    """
    lon = da.longitude.values.astype(float)
    lat = da.latitude.values.astype(float)
    source_depth = da.values.astype(float)

    if lon.size < 2 or lat.size < 2:
        raise ValueError("source crop is too small for coarsening")

    source_dx_all = np.diff(lon)
    source_dy_all = np.diff(lat)

    source_dx = float(np.median(source_dx_all))
    source_dy = float(np.median(source_dy_all))

    dx_tolerance = max(abs(source_dx), 1.0) * 1.0e-4
    dy_tolerance = max(abs(source_dy), 1.0) * 1.0e-4

    if not np.allclose(
        source_dx_all,
        source_dx,
        rtol=0.0,
        atol=dx_tolerance,
    ):
        raise ValueError("source longitude spacing is not regular")

    if not np.allclose(
        source_dy_all,
        source_dy,
        rtol=0.0,
        atol=dy_tolerance,
    ):
        raise ValueError("source latitude spacing is not regular")

    ratio_x = dx / source_dx
    ratio_y = dy / source_dy

    block_x = int(round(ratio_x))
    block_y = int(round(ratio_y))

    if not np.isclose(
        ratio_x,
        block_x,
        rtol=0.0,
        atol=1.0e-3,
    ):
        raise ValueError(
            f"DX/source_dx must be an integer, got {ratio_x:.10f}"
        )

    if not np.isclose(
        ratio_y,
        block_y,
        rtol=0.0,
        atol=1.0e-3,
    ):
        raise ValueError(
            f"DY/source_dy must be an integer, got {ratio_y:.10f}"
        )

    nx = x_centers.size
    ny = y_centers.size

    # Actual model-cell outer edges.
    xlo = x_centers[0] - dx / 2.0
    xhi = x_centers[-1] + dx / 2.0
    ylo = y_centers[0] - dy / 2.0
    yhi = y_centers[-1] + dy / 2.0

    ii = np.flatnonzero(
        (lon >= xlo - COORD_TOL)
        & (lon < xhi - COORD_TOL)
    )
    jj = np.flatnonzero(
        (lat >= ylo - COORD_TOL)
        & (lat < yhi - COORD_TOL)
    )

    if ii.size == 0 or jj.size == 0:
        raise ValueError(
            "no source-grid cells fall inside the model domain"
        )

    source_depth = source_depth[np.ix_(jj, ii)]

    expected_shape = (
        ny * block_y,
        nx * block_x,
    )

    if source_depth.shape != expected_shape:
        raise ValueError(
            "source grid does not align with the model grid:\n"
            f"  selected source shape: {source_depth.shape}\n"
            f"  expected shape:        {expected_shape}\n"
            f"  source dx/dy:           {source_dx}, {source_dy}\n"
            f"  block size:             {block_x} x {block_y}\n"
            f"  selected lon range:     {lon[ii[0]]}, {lon[ii[-1]]}\n"
            f"  selected lat range:     {lat[jj[0]]}, {lat[jj[-1]]}"
        )

    # Convert to:
    # (model_y, model_x, source_y_in_cell, source_x_in_cell)
    blocks = source_depth.reshape(
        ny,
        block_y,
        nx,
        block_x,
    ).transpose(0, 2, 1, 3)

    source_wet = (
        np.isfinite(blocks)
        & (blocks > 0.0)
    )

    wet_count = source_wet.sum(axis=(2, 3))
    source_count = block_x * block_y
    wet_fraction = wet_count / source_count

    depth_sum = np.where(
        source_wet,
        blocks,
        0.0,
    ).sum(axis=(2, 3))

    depth_wet_only = np.full(
        (ny, nx),
        np.nan,
        dtype=float,
    )

    np.divide(
        depth_sum,
        wet_count,
        out=depth_wet_only,
        where=wet_count > 0,
    )

    depth_area_mean = depth_sum / source_count

    # Geometric compromise between wet-only and area-mean depth.
    depth = np.sqrt(
        depth_wet_only * depth_area_mean
    )
    depth[wet_count == 0] = np.nan

    diagnostics = {
        "source_dx": source_dx,
        "source_dy": source_dy,
        "block_x": block_x,
        "block_y": block_y,
        "selected_source_shape": source_depth.shape,
        "selected_lon_range": (
            float(lon[ii[0]]),
            float(lon[ii[-1]]),
        ),
        "selected_lat_range": (
            float(lat[jj[0]]),
            float(lat[jj[-1]]),
        ),
    }

    return depth, wet_fraction, diagnostics


# ============================================================
# Binary output
# ============================================================
def write_and_verify_bathymetry(path, bathymetry):
    """
    Write big-endian float32 MITgcm bathymetry and read it back to verify
    the output file.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bathymetry_be = np.asarray(
        bathymetry,
        dtype=">f4",
    )
    bathymetry_be.tofile(path)

    expected_size = (
        bathymetry_be.size
        * bathymetry_be.dtype.itemsize
    )
    actual_size = path.stat().st_size

    if actual_size != expected_size:
        raise IOError(
            f"binary file size is {actual_size} bytes, "
            f"expected {expected_size} bytes"
        )

    restored = np.fromfile(
        path,
        dtype=">f4",
    ).reshape(bathymetry_be.shape)

    expected = bathymetry_be.astype(
        np.float32,
    )

    if not np.array_equal(restored, expected):
        raise IOError(
            "bathymetry binary read-back verification failed"
        )


# ============================================================
# Main workflow
# ============================================================
def main():
    nx_float = XRG / DX
    ny_float = YRG / DY

    nx = int(round(nx_float))
    ny = int(round(ny_float))

    if not np.isclose(nx_float, nx):
        raise ValueError(
            "XRG must be an integer multiple of DX"
        )

    if not np.isclose(ny_float, ny):
        raise ValueError(
            "YRG must be an integer multiple of DY"
        )

    x_centers = X0 + np.arange(nx) * DX
    y_centers = Y0 + np.arange(ny) * DY

    x1 = x_centers[-1]
    y1 = y_centers[-1]

    # Crop only to the actual model-grid outer edges.
    # The tiny tolerance handles floating-point coordinate comparisons;
    # it is not a physical source-data buffer.
    xll = x_centers[0] - DX / 2.0 - COORD_TOL
    xrr = x_centers[-1] + DX / 2.0 + COORD_TOL
    yd = y_centers[0] - DY / 2.0 - COORD_TOL
    yu = y_centers[-1] + DY / 2.0 + COORD_TOL

    source_depth = open_deptho(
        FBATH,
        xll=xll,
        xrr=xrr,
        yd=yd,
        yu=yu,
    )

    # --------------------------------------------------------
    # High-resolution connectivity
    # --------------------------------------------------------
    # Connectivity is evaluated only inside the actual model domain.
    # A region connected to the main ocean only outside the model domain
    # is therefore treated as disconnected.
    source_wet_original = (
        np.isfinite(source_depth.values)
        & (source_depth.values > 0.0)
    )

    source_main_ocean = largest_connected_region(
        source_wet_original
    )

    source_wet_removed = (
        source_wet_original
        & ~source_main_ocean
    )

    source_depth = source_depth.where(
        source_main_ocean
    )

    # --------------------------------------------------------
    # Coarsening
    # --------------------------------------------------------
    depth, wet_fraction, coarsen_info = (
        coarsen_depth_to_model_grid(
            source_depth,
            x_centers=x_centers,
            y_centers=y_centers,
            dx=DX,
            dy=DY,
        )
    )

    # Initial model-grid land/sea decision.
    wet_before_connectivity = (
        wet_fraction >= WET_FRAC_THRESHOLD
    )

    # Ignore connections that exist only along the outermost model ring.
    wet = largest_region_ignoring_outer_ring(
        wet_before_connectivity
    )

    model_wet_removed = (
        wet_before_connectivity
        & ~wet
    )

    # --------------------------------------------------------
    # Final depth
    # --------------------------------------------------------
    depth = np.where(
        wet,
        depth,
        np.nan,
    )

    shallow_cells = (
        wet
        & (depth < MINIMUM_DEPTH)
    )
    depth[shallow_cells] = MINIMUM_DEPTH

    # MITgcm convention:
    # ocean = negative depth; land = zero.
    bathymetry = np.zeros(
        (ny, nx),
        dtype=float,
    )
    bathymetry[wet] = -depth[wet]

    if not np.all(np.isfinite(bathymetry)):
        raise ValueError(
            "final bathymetry contains non-finite values"
        )

    if np.any(bathymetry > 0.0):
        raise ValueError(
            "final bathymetry contains positive values"
        )

    write_and_verify_bathymetry(
        OUT_BATHY,
        bathymetry,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    print(f"source file: {FBATH}")
    print(f"source crop shape: {source_depth.shape}")
    print(f"target shape: {bathymetry.shape}")
    print(f"expected target shape: {(ny, nx)}")

    print(
        "target longitude centres:",
        f"{X0:.3f} to {x1:.3f}",
    )
    print(
        "target latitude centres:",
        f"{Y0:.3f} to {y1:.3f}",
    )
    print(
        "model outer edges:",
        f"{X0 - DX / 2:.3f}, "
        f"{x1 + DX / 2:.3f}, "
        f"{Y0 - DY / 2:.3f}, "
        f"{y1 + DY / 2:.3f}",
    )

    print(
        "source dx/dy:",
        coarsen_info["source_dx"],
        coarsen_info["source_dy"],
    )
    print(
        "coarsening block:",
        coarsen_info["block_x"],
        "x",
        coarsen_info["block_y"],
    )
    print(
        "selected source shape:",
        coarsen_info["selected_source_shape"],
    )
    print(
        "selected source longitude:",
        coarsen_info["selected_lon_range"],
    )
    print(
        "selected source latitude:",
        coarsen_info["selected_lat_range"],
    )

    print(
        "removed disconnected source wet cells:",
        int(source_wet_removed.sum()),
    )
    print(
        "removed disconnected model wet cells:",
        int(model_wet_removed.sum()),
    )
    print(
        "wet-fraction range:",
        float(np.nanmin(wet_fraction)),
        float(np.nanmax(wet_fraction)),
    )
    print(
        "wet model cells:",
        int(wet.sum()),
        "/",
        wet.size,
    )
    print(
        "deepened shallow model cells:",
        int(shallow_cells.sum()),
    )

    if wet.any():
        print(
            "minimum/maximum wet depth:",
            float(depth[wet].min()),
            float(depth[wet].max()),
        )

    print(f"wrote and verified: {OUT_BATHY}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    if MAKE_PLOT:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            constrained_layout=True,
        )

        im0 = axes[0].pcolormesh(
            x_centers,
            y_centers,
            wet_fraction,
            shading="auto",
            vmin=0.0,
            vmax=1.0,
        )

        axes[0].contour(
            x_centers,
            y_centers,
            wet.astype(np.uint8),
            levels=[0.5],
            linewidths=0.7,
        )

        axes[0].set_title(
            "Source wet fraction and final model coastline"
        )
        axes[0].set_ylabel("Latitude")

        fig.colorbar(
            im0,
            ax=axes[0],
            label="Wet fraction",
        )

        im1 = axes[1].pcolormesh(
            x_centers,
            y_centers,
            depth,
            shading="auto",
        )

        axes[1].set_title(
            "Final positive model depth"
        )
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")

        fig.colorbar(
            im1,
            ax=axes[1],
            label="Depth (m)",
        )

        fig.savefig(
            OUT_FIGURE,
            dpi=150,
        )
        plt.close(fig)

        print(f"wrote: {OUT_FIGURE}")


if __name__ == "__main__":
    main()