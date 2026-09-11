"""
Shared utilities: slicing, filtering, colormap helpers, etc.
All plotters import from here to stay DRY.
"""

from __future__ import annotations
import numpy as np
import xarray as xr
from typing import Optional


# ── dataset slicing ────────────────────────────────────────────────────────────

def sel_nearest(ds: xr.Dataset, dim: str, value: float) -> xr.Dataset:
    """Select the nearest index along *dim* to *value*."""
    return ds.sel({dim: value}, method="nearest")


def isel_depth_range(ds: xr.Dataset,
                     z_dim: str = "Z",
                     z_min: float = 0,
                     z_max: float = 500) -> xr.Dataset:
    """Slice dataset to a depth range [z_min, z_max] (both in metres, positive down)."""
    z = ds[z_dim].values
    mask = (np.abs(z) >= z_min) & (np.abs(z) <= z_max)
    return ds.isel({z_dim: mask})


def bandpass_filter(da: xr.DataArray,
                    low_period_days: float,
                    high_period_days: float,
                    time_dim: str = "time") -> xr.DataArray:
    """
    Simple Lanczos-style bandpass via FFT zeroing.

    For year-to-interannual work (periods >> 30 days) this is sufficient;
    replace with scipy.signal.butter + filtfilt for production use.

    Parameters
    ----------
    low_period_days  : retain frequencies *slower* than this (longer periods)
    high_period_days : retain frequencies *faster* than this (shorter periods)
    """
    import scipy.signal as sig

    dt_days = float((da[time_dim][1] - da[time_dim][0]) / np.timedelta64(1, "D"))
    fs = 1.0 / dt_days  # cycles per day

    low_freq  = 1.0 / low_period_days   # high-pass cutoff
    high_freq = 1.0 / high_period_days  # low-pass cutoff

    nyq = 0.5 * fs
    low  = low_freq  / nyq
    high = high_freq / nyq

    # 4th-order Butterworth bandpass
    b, a = sig.butter(4, [low, high], btype="band")

    def _filt(arr):
        return sig.filtfilt(b, a, arr)

    return xr.apply_ufunc(
        _filt, da,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
    )


# ── coordinate helpers ─────────────────────────────────────────────────────────

def get_z_coord(ds: xr.Dataset) -> str:
    """Guess the name of the vertical coordinate."""
    for candidate in ("Z", "depth", "lev", "z_t", "z_rho", "ZC"):
        if candidate in ds.coords or candidate in ds:
            return candidate
    raise ValueError("Cannot find vertical coordinate. Pass z_dim explicitly.")


def get_time_coord(ds: xr.Dataset) -> str:
    for candidate in ("time", "Time", "T"):
        if candidate in ds.coords:
            return candidate
    raise ValueError("Cannot find time coordinate.")


# ── colormap & normalisation ───────────────────────────────────────────────────

def symmetric_norm(data: np.ndarray, pct: float = 99.0):
    """Return a matplotlib Normalize centred on zero, scaled to *pct* percentile."""
    import matplotlib.colors as mcolors
    vmax = np.nanpercentile(np.abs(data), pct)
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def make_levels(data: np.ndarray, n: int = 20, symmetric: bool = True):
    if symmetric:
        vmax = np.nanpercentile(np.abs(data), 99)
        return np.linspace(-vmax, vmax, n)
    return np.linspace(np.nanmin(data), np.nanmax(data), n)


# ── figure annotation ──────────────────────────────────────────────────────────

def add_colorbar(fig, ax, mappable, label: str = "", **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cb = fig.colorbar(mappable, cax=cax, **kwargs)
    cb.set_label(label, fontsize=9)
    return cb


def label_axes(ax,
               title: str = "",
               xlabel: str = "",
               ylabel: str = "",
               fontsize: int = 10):
    ax.set_title(title, fontsize=fontsize, pad=6)
    ax.set_xlabel(xlabel, fontsize=fontsize - 1)
    ax.set_ylabel(ylabel, fontsize=fontsize - 1)
    ax.tick_params(labelsize=fontsize - 2)
