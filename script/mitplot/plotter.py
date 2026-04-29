"""
mitplot.plotter
====================
High-level public API.  Users only need to interact with this module.

Quick start
-----------
    from mitplot import DiagPlotter

    dp = DiagPlotter(ds_model, ds_ref=ds_reanalysis)

    # single plot
    fig, ax = dp.plot("hovmoller_u", lat=0, depth=100)

    # multi-panel figure
    fig = dp.panel(
        [
            dict(kind="hovmoller_u",       lat=0, depth=100),
            dict(kind="section_u_z",       lat=0, z_max=400),
            dict(kind="timeseries_euc",    lon=140),
            dict(kind="spectrum_u",        lon=140, depth=100,
                 mark_periods=[1, 3, 5]),
        ],
        ncols=2,
        figsize=(14, 10),
        suptitle="Equatorial Pacific  –  Model vs Reanalysis",
    )
    fig.savefig("diagnostics.pdf", dpi=150, bbox_inches="tight")

    # see what's available
    dp.list_available()
"""

from __future__ import annotations
import importlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from typing import Any

from .registry import get_plotter, list_plotters
from .utils.common import add_colorbar

# ── auto-import all built-in plotters so they self-register ───────────────────
importlib.import_module("mitplot.plotters.ocean_plots")


class MitPlotter:
    """
    Facade that couples a dataset pair to the plot registry.

    Parameters
    ----------
    ds : xr.Dataset
        Your MITgcm output.
    ds_ref : xr.Dataset, optional
        Reanalysis / reference dataset for difference and overlay plots.
    defaults : dict, optional
        Default keyword arguments forwarded to every plotter
        (e.g. ``{"lat_dim": "lat", "lon_dim": "lon"}``).
    """

    def __init__(self,
                 ds: xr.Dataset,
                 ds_ref: xr.Dataset | None = None,
                 defaults: dict[str, Any] | None = None):
        self.ds      = ds
        self.ds_ref  = ds_ref
        self._defaults = defaults or {}

    # ── single-panel ──────────────────────────────────────────────────────────

    def plot(self,
             kind: str,
             *,
             ax: plt.Axes | None = None,
             figsize: tuple = (9, 4),
             colorbar: bool = True,
             colorbar_label: str = "",
             save: str | None = None,
             **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """
        Draw a single diagnostic plot.

        Parameters
        ----------
        kind     : registered plot name (see ``list_available()``)
        ax       : existing Axes to draw into; a new figure is created if None
        colorbar : whether to attach a colorbar (only for contourf-type plots)
        save     : file path to save the figure (optional)
        **kwargs : forwarded to the plotter function

        Returns
        -------
        fig, ax
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize,
                                   constrained_layout=True)
        else:
            fig = ax.figure

        merged = {**self._defaults, **kwargs}

        # inject ds_ref only when the plotter accepts it
        if self.ds_ref is not None and "ds_ref" not in merged:
            merged["ds_ref"] = self.ds_ref

        fn = get_plotter(kind)

        # section_u_z_diff / map_sst_bias take ds_model as first positional arg
        result = fn(self.ds, ax, **merged)

        if colorbar and result is not None:
            add_colorbar(fig, ax, result, label=colorbar_label)

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")

        return fig, ax

    # ── multi-panel ───────────────────────────────────────────────────────────

    def panel(self,
              specs: list[dict[str, Any]],
              *,
              ncols: int = 2,
              figsize: tuple | None = None,
              suptitle: str = "",
              colorbar: bool = True,
              save: str | None = None,
              hspace: float = 0.45,
              wspace: float = 0.35) -> plt.Figure:
        """
        Lay out multiple diagnostic plots in a grid.

        Parameters
        ----------
        specs  : list of dicts, each must have a ``"kind"`` key plus any
                 kwargs for that plotter.
        ncols  : number of columns in the grid.
        save   : file path to save the figure (optional).

        Example
        -------
        dp.panel([
            dict(kind="hovmoller_u", lat=0, depth=100),
            dict(kind="section_u_z", lat=0),
            dict(kind="spectrum_u",  lon=140, mark_periods=[1,3,5]),
        ], ncols=2, suptitle="Overview")
        """
        n     = len(specs)
        nrows = (n + ncols - 1) // ncols
        fw, fh = figsize or (6.5 * ncols, 4.5 * nrows)

        fig = plt.figure(figsize=(fw, fh))
        gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                                hspace=hspace, wspace=wspace)

        for i, spec in enumerate(specs):
            spec = dict(spec)                      # don't mutate caller's dict
            kind = spec.pop("kind")
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col])

            merged = {**self._defaults, **spec}
            if self.ds_ref is not None and "ds_ref" not in merged:
                merged["ds_ref"] = self.ds_ref

            fn = get_plotter(kind)
            result = fn(self.ds, ax, **merged)

            if colorbar and result is not None:
                add_colorbar(fig, ax, result)

        if suptitle:
            fig.suptitle(suptitle, fontsize=13, y=1.01, fontweight="bold")

        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")

        return fig

    # ── introspection ─────────────────────────────────────────────────────────

    def list_available(self) -> list[str]:
        """Print and return all registered plot names."""
        names = list_plotters()
        print("Available plot types:")
        for n in names:
            fn = get_plotter(n)
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {n:<30} - {doc}")

    def __repr__(self) -> str:
        ref_str = "with reanalysis ref" if self.ds_ref is not None else "no ref"
        return (f"DiagPlotter({ref_str}, "
                f"{len(list_plotters())} plotters registered)")
