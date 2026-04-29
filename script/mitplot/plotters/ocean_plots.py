"""
Concrete plot functions for MITgcm diagnostics.

Every function here is registered into the global registry via
``@register_plot``.  Adding a new plot type = writing one function here
(or in any module that imports and calls ``register_plot``).

Registered names
----------------
  hovmoller_u          – longitude-time Hovmöller of zonal velocity
  section_u_z          – depth-longitude section of mean zonal velocity
  section_u_z_diff     – depth-longitude difference (model minus reanalysis)
  timeseries_euc       – EUC core speed time series
  spectrum_u           – power spectral density of zonal velocity
  vertical_profile_u   – mean vertical profile at a point
  map_sst_bias         – surface temperature bias map
  bandpass_hov         – Hovmöller of bandpass-filtered zonal velocity
"""

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


# ── 1. Hovmöller: longitude × time ────────────────────────────────────────────

@register_plot("hovmoller_u")
def plot_hovmoller_u(ds: xr.Dataset,
                     ax: plt.Axes,
                     *,
                     var: str = "UVEL",
                     lat: float = 0.0,
                     depth: float = 100.0,
                     lat_dim: str = "YC",
                     lon_dim: str = "XC",
                     z_dim: str | None = None,
                     time_dim: str | None = None,
                     cmap: str = "RdBu_r",
                     n_levels: int = 21,
                     title: str | None = None,
                     **_) -> plt.cm.ScalarMappable:
    """
    Longitude–time Hovmöller of zonal velocity at a fixed latitude and depth.

    Parameters
    ----------
    lat   : latitude to slice (nearest)
    depth : depth level in metres (nearest, positive down)
    """
    z_dim   = z_dim   or get_z_coord(ds)
    time_dim = time_dim or get_time_coord(ds)

    u = _require_var(ds, var)
    u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]
    u = sel_nearest(u.to_dataset(name=var), z_dim, depth)[var]

    lons  = u[lon_dim].values
    times = u[time_dim].values
    data  = u.values          # shape: (time, lon)

    levels = make_levels(data, n=n_levels, symmetric=True)
    cf = ax.contourf(lons, np.arange(len(times)), data,
                     levels=levels, cmap=cmap, extend="both")

    # y-axis: show years
    _set_time_yticks(ax, times, time_dim)

    label_axes(ax,
               title=title or f"Hovmöller {var}  lat={lat}°  z={depth} m",
               xlabel="Longitude (°E)",
               ylabel="Time")
    return cf


# ── 2. Depth × longitude section ──────────────────────────────────────────────

@register_plot("section_u_z")
def plot_section_u_z(ds: xr.Dataset,
                     ax: plt.Axes,
                     *,
                     var: str = "UVEL",
                     lat: float = 0.0,
                     lat_dim: str = "YC",
                     lon_dim: str = "XC",
                     z_dim: str | None = None,
                     z_max: float = 400.0,
                     cmap: str = "RdBu_r",
                     n_levels: int = 21,
                     time_mean: bool = True,
                     title: str | None = None,
                     **_) -> plt.cm.ScalarMappable:
    """Depth × longitude section of zonal velocity (time-mean by default)."""
    z_dim = z_dim or get_z_coord(ds)

    u = _require_var(ds, var)
    u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]

    if time_mean and get_time_coord(ds) in u.dims:
        u = u.mean(get_time_coord(ds))

    u = isel_depth_range(u.to_dataset(name=var), z_dim, 0, z_max)[var]

    lons   = u[lon_dim].values
    depths = np.abs(u[z_dim].values)
    data   = u.values   # (z, lon)

    levels = make_levels(data, n=n_levels, symmetric=True)
    cf = ax.contourf(lons, depths, data, levels=levels, cmap=cmap, extend="both")
    ax.contour(lons, depths, data, levels=[0], colors="k", linewidths=0.8)
    ax.set_ylim(z_max, 0)

    label_axes(ax,
               title=title or f"Section {var}  lat={lat}°",
               xlabel="Longitude (°E)",
               ylabel="Depth (m)")
    return cf


# ── 3. Model − reanalysis difference section ──────────────────────────────────

@register_plot("section_u_z_diff")
def plot_section_u_z_diff(ds_model: xr.Dataset,
                           ax: plt.Axes,
                           *,
                           ds_ref: xr.Dataset,
                           var: str = "UVEL",
                           lat: float = 0.0,
                           lat_dim: str = "YC",
                           lon_dim: str = "XC",
                           z_dim: str | None = None,
                           z_max: float = 400.0,
                           cmap: str = "RdBu_r",
                           n_levels: int = 21,
                           title: str | None = None,
                           **_) -> plt.cm.ScalarMappable:
    """
    Difference plot: model minus reanalysis in depth × longitude.

    Parameters
    ----------
    ds_ref : reference (reanalysis) dataset – must share coordinates with ds_model
    """
    z_dim = z_dim or get_z_coord(ds_model)

    def _extract(ds):
        u = _require_var(ds, var)
        u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]
        td = get_time_coord(ds)
        if td in u.dims:
            u = u.mean(td)
        u = isel_depth_range(u.to_dataset(name=var), z_dim, 0, z_max)[var]
        return u

    diff = _extract(ds_model) - _extract(ds_ref)

    lons   = diff[lon_dim].values
    depths = np.abs(diff[z_dim].values)
    data   = diff.values

    levels = make_levels(data, n=n_levels, symmetric=True)
    cf = ax.contourf(lons, depths, data, levels=levels, cmap=cmap, extend="both")
    ax.set_ylim(z_max, 0)

    label_axes(ax,
               title=title or f"Δ{var} (model − ref)  lat={lat}°",
               xlabel="Longitude (°E)",
               ylabel="Depth (m)")
    return cf


# ── 4. EUC core speed time series ─────────────────────────────────────────────

@register_plot("timeseries_euc")
def plot_timeseries_euc(ds: xr.Dataset,
                         ax: plt.Axes,
                         *,
                         var: str = "UVEL",
                         lat: float = 0.0,
                         lon: float = 140.0,
                         lat_dim: str = "YC",
                         lon_dim: str = "XC",
                         z_dim: str | None = None,
                         z_euc_range: tuple = (50, 300),
                         ds_ref: xr.Dataset | None = None,
                         ref_label: str = "Reanalysis",
                         model_label: str = "Model",
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
        u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]
        u = sel_nearest(u.to_dataset(name=var), lon_dim, lon)[var]
        u = isel_depth_range(u.to_dataset(name=var), z_dim, *z_euc_range)[var]
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


# ── 5. Power spectral density ──────────────────────────────────────────────────

@register_plot("spectrum_u")
def plot_spectrum_u(ds: xr.Dataset,
                    ax: plt.Axes,
                    *,
                    var: str = "UVEL",
                    lat: float = 0.0,
                    lon: float = 140.0,
                    depth: float = 100.0,
                    lat_dim: str = "YC",
                    lon_dim: str = "XC",
                    z_dim: str | None = None,
                    ds_ref: xr.Dataset | None = None,
                    ref_label: str = "Reanalysis",
                    model_label: str = "Model",
                    mark_periods: list[float] | None = None,
                    title: str | None = None,
                    **_):
    """
    Power spectral density of U at a point.
    X-axis in years; vertical lines mark periods of interest.

    mark_periods : list of periods (years) to annotate (e.g. [1, 3, 5])
    """
    from scipy import signal as sig

    z_dim    = z_dim    or get_z_coord(ds)
    time_dim = get_time_coord(ds)

    def _extract_ts(dataset):
        u = _require_var(dataset, var)
        u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]
        u = sel_nearest(u.to_dataset(name=var), lon_dim, lon)[var]
        u = sel_nearest(u.to_dataset(name=var), z_dim, depth)[var]
        return u

    def _psd(da):
        dt_days = float(
            (da[time_dim][1] - da[time_dim][0]) / np.timedelta64(1, "D")
        )
        fs = 1.0 / dt_days   # cycles / day
        x  = da.values - da.values.mean()
        f, pxx = sig.welch(x, fs=fs, nperseg=min(256, len(x) // 2))
        period_years = 1.0 / (f[1:] * 365.25)   # skip DC
        return period_years, pxx[1:]

    p_m, pxx_m = _psd(_extract_ts(ds))
    ax.loglog(p_m, pxx_m, lw=1.5, color="#1f6feb", label=model_label)

    if ds_ref is not None:
        p_r, pxx_r = _psd(_extract_ts(ds_ref))
        ax.loglog(p_r, pxx_r, lw=1.5, color="#cf4a30", ls="--", label=ref_label)
        ax.legend(fontsize=8, framealpha=0.4)

    # mark target periods
    for p in (mark_periods or [1, 3, 5]):
        ax.axvline(p, color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.text(p, ax.get_ylim()[1], f"{p}yr", fontsize=7,
                ha="center", va="bottom", color="gray")

    ax.set_xlim(left=0.1)
    label_axes(ax,
               title=title or f"PSD {var}  {lon}°E lat={lat}° z={depth} m",
               xlabel="Period (years)",
               ylabel="PSD (m² s⁻² / cpd)")


# ── 6. Vertical profile at a point ────────────────────────────────────────────

@register_plot("vertical_profile_u")
def plot_vertical_profile_u(ds: xr.Dataset,
                             ax: plt.Axes,
                             *,
                             var: str = "UVEL",
                             lat: float = 0.0,
                             lon: float = 140.0,
                             lat_dim: str = "YC",
                             lon_dim: str = "XC",
                             z_dim: str | None = None,
                             z_max: float = 400.0,
                             ds_ref: xr.Dataset | None = None,
                             ref_label: str = "Reanalysis",
                             model_label: str = "Model",
                             title: str | None = None,
                             **_):
    """Mean vertical profile of U at a fixed lon/lat point."""
    z_dim    = z_dim    or get_z_coord(ds)
    time_dim = get_time_coord(ds)

    def _profile(dataset):
        u = _require_var(dataset, var)
        u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]
        u = sel_nearest(u.to_dataset(name=var), lon_dim, lon)[var]
        if time_dim in u.dims:
            u = u.mean(time_dim)
        u = isel_depth_range(u.to_dataset(name=var), z_dim, 0, z_max)[var]
        return u

    prof_m = _profile(ds)
    depths  = np.abs(prof_m[z_dim].values)

    ax.plot(prof_m.values, depths, lw=1.8, color="#1f6feb", label=model_label)

    if ds_ref is not None:
        prof_r = _profile(ds_ref)
        ax.plot(prof_r.values, np.abs(prof_r[z_dim].values),
                lw=1.8, color="#cf4a30", ls="--", label=ref_label)
        ax.legend(fontsize=8, framealpha=0.4)

    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_ylim(z_max, 0)
    label_axes(ax,
               title=title or f"Profile {var}  {lon}°E lat={lat}°",
               xlabel="U (m s⁻¹)",
               ylabel="Depth (m)")


# ── 7. SST bias map ───────────────────────────────────────────────────────────

@register_plot("map_sst_bias")
def plot_map_sst_bias(ds: xr.Dataset,
                       ax: plt.Axes,
                       *,
                       var: str = "THETA",
                       ds_ref: xr.Dataset,
                       lat_dim: str = "YC",
                       lon_dim: str = "XC",
                       z_dim: str | None = None,
                       cmap: str = "RdBu_r",
                       n_levels: int = 21,
                       title: str | None = None,
                       **_) -> plt.cm.ScalarMappable:
    """Surface temperature bias map (model − reanalysis, top-level mean)."""
    z_dim    = z_dim    or get_z_coord(ds)
    time_dim = get_time_coord(ds)

    def _sst(dataset):
        t = _require_var(dataset, var)
        t = t.isel({z_dim: 0})
        td = get_time_coord(dataset)
        if td in t.dims:
            t = t.mean(td)
        return t

    bias = _sst(ds) - _sst(ds_ref)
    data  = bias.values
    lons  = bias[lon_dim].values
    lats  = bias[lat_dim].values

    levels = make_levels(data, n=n_levels, symmetric=True)
    cf = ax.contourf(lons, lats, data, levels=levels, cmap=cmap, extend="both")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)

    label_axes(ax,
               title=title or f"SST bias (model − ref)  [{var}]",
               xlabel="Longitude (°E)",
               ylabel="Latitude (°N)")
    return cf


# ── 8. Band-pass filtered Hovmöller ──────────────────────────────────────────

@register_plot("bandpass_hov")
def plot_bandpass_hov(ds: xr.Dataset,
                       ax: plt.Axes,
                       *,
                       var: str = "UVEL",
                       lat: float = 0.0,
                       depth: float = 100.0,
                       lat_dim: str = "YC",
                       lon_dim: str = "XC",
                       z_dim: str | None = None,
                       low_period_days: float = 300.0,
                       high_period_days: float = 2600.0,
                       cmap: str = "RdBu_r",
                       n_levels: int = 21,
                       title: str | None = None,
                       **_) -> plt.cm.ScalarMappable:
    """
    Hovmöller of band-pass filtered zonal velocity.

    Default band: 300 – 2600 days ≈ annual-to-interannual.
    """
    z_dim    = z_dim    or get_z_coord(ds)
    time_dim = get_time_coord(ds)

    u = _require_var(ds, var)
    u = sel_nearest(u.to_dataset(name=var), lat_dim, lat)[var]
    u = sel_nearest(u.to_dataset(name=var), z_dim, depth)[var]

    u_filt = bandpass_filter(u, low_period_days, high_period_days, time_dim)

    lons   = u_filt[lon_dim].values
    times  = u_filt[time_dim].values
    data   = u_filt.values

    levels = make_levels(data, n=n_levels, symmetric=True)
    cf = ax.contourf(lons, np.arange(len(times)), data,
                     levels=levels, cmap=cmap, extend="both")
    _set_time_yticks(ax, times, time_dim)

    label_axes(ax,
               title=title or (f"Bandpass Hovmöller {var}  "
                               f"{low_period_days:.0f}–{high_period_days:.0f} d"),
               xlabel="Longitude (°E)",
               ylabel="Time")
    return cf


# ── private helpers ────────────────────────────────────────────────────────────

def _set_time_yticks(ax, times, time_dim, max_ticks: int = 8):
    """Replace numeric y-axis with readable year labels."""
    n = len(times)
    step = max(1, n // max_ticks)
    idxs = np.arange(0, n, step)
    try:
        labels = [str(np.datetime64(times[i], "Y")) for i in idxs]
    except Exception:
        labels = [str(i) for i in idxs]
    ax.set_yticks(idxs)
    ax.set_yticklabels(labels, fontsize=8)
