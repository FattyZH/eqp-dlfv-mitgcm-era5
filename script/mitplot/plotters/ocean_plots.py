from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr

from mitplot.registry import register_plot
from mitplot.utils.common import (
    sel_nearest, isel_depth_range, bandpass_filter,
    get_z_coord, get_time_coord,
    symmetric_norm, make_levels, add_colorbar, label_axes,
)


# ── helpers private to this module ────────────────────────────────────────────

def _require_var(ds: xr.Dataset, var: str) -> xr.DataArray:
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset. "
                       f"Available: {list(ds.data_vars)}")
    return ds[var]


@register_plot("timeseries_euc")
def plot_timeseries_euc(ds: xr.Dataset,
                         ax: plt.Axes,
                         *,
                         var: str = "UVEL",
                         lon: float = 140.0,
                         lat_dim: str = "YC",
                         lon_dim: str = "XG",
                         z_dim: str | None = None,
                         lat_range: float = (-2,2),
                         z_range: tuple = (50, 300),
                         title: str | None = None,
                         **_):
    """
    Time series of EUC core speed (max zonal velocity in depth range).

    Optionally overlays reference dataset.
    """
    z_dim   = z_dim   or get_z_coord(ds)
    time_dim = get_time_coord(ds)

    def _euc_core(dataset):
        u = _require_var(dataset, var)
        u = u.sel(lon_dim=lon)
        u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]
        u = isel_depth_range(u.to_dataset(name=var), z_dim, *z_range)[var]
        return u.max(z_dim)   # EUC core = depth-max of U

    ts_model = _euc_core(ds)
    times = ts_model[time_dim].values

    ax.plot(times, ts_model.values, lw=1.5, color="#1f6feb", label=model_label)

    if ds_ref is not None:
        ts_ref = _euc_core(ds_ref)
        ax.plot(ts_ref[get_time_coord(ds_ref)].values, ts_ref.values,
                lw=1.5, color="#cf4a30", ls="--", label=ref_label)
        ax.legend(fontsize=8, framealpha=0.4)

    ax.axhline(0, color="k", lw=0.5, ls=":")
    label_axes(ax,
               title=title or f"EUC core speed  {lon}°E, lat={lat}°",
               xlabel="Time",
               ylabel="U (m s⁻¹)")
    ax.xaxis.set_major_locator(mticker.AutoLocator())
