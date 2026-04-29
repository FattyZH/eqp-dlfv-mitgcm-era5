"""
example_usage.py
================
Demonstrates every pattern in mitplot.

Run as:
    python example_usage.py
(uses synthetic data, no real MITgcm output required)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ── 0. build a tiny synthetic dataset ─────────────────────────────────────────

def make_fake_ds(seed=0, n_time=120, n_lon=60, n_lat=20, n_z=30, prefix=""):
    rng = np.random.default_rng(seed)
    times = np.array(
        [np.datetime64("2000-01") + np.timedelta64(i * 30, "D")
         for i in range(n_time)]
    )
    lons   = np.linspace(120, 280, n_lon)
    lats   = np.linspace(-10, 10, n_lat)
    depths = np.linspace(0, 500, n_z)

    # synthetic EUC-like signal: westward surface, eastward core ~100 m
    z_prof = np.exp(-((depths - 100) / 60) ** 2) - 0.3 * np.exp(-depths / 30)
    base   = (
        z_prof[None, None, :, None]                           # (1,1,z,1)
        * np.cos(np.linspace(0, np.pi, n_lon))[None, None, None, :]  # lon shape
    )
    noise  = 0.05 * rng.standard_normal((n_time, n_lat, n_z, n_lon))
    UVEL   = (base + noise).transpose(0, 1, 2, 3)  # (time, lat, z, lon)

    return xr.Dataset(
        {
            "UVEL":  (["time", "YC", "Z", "XC"], UVEL),
            "THETA": (["time", "YC", "Z", "XC"],
                      25.0 - depths[None, None, :, None] * 0.04
                      + 0.3 * rng.standard_normal((n_time, n_lat, n_z, n_lon))),
        },
        coords={"time": times, "YC": lats, "Z": -depths, "XC": lons},
    )


ds_model = make_fake_ds(seed=0)
ds_ref   = make_fake_ds(seed=42)   # "reanalysis" – slightly different noise


# ── 1. Create the plotter ──────────────────────────────────────────────────────

from mitplot import MitPlotter

dp = MitPlotter(
    ds_model,
    ds_ref=ds_ref,
    defaults={"lat_dim": "YC", "lon_dim": "XC", "z_dim": "Z"},
)

print(dp)
print()
dp.list_available()
print()


# ── 2. Single plot ─────────────────────────────────────────────────────────────

fig, ax = dp.plot(
    "hovmoller_u",
    lat=0, depth=-100,
    figsize=(10, 5),
    title="Synthetic Hovmöller – zonal velocity at 100 m",
)
plt.savefig("single_hovmoller.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: single_hovmoller.png")


# ── 3. Multi-panel overview ────────────────────────────────────────────────────

fig = dp.panel(
    [
        dict(kind="hovmoller_u",
             lat=0, depth=-100,
             title="Hovmöller U  (100 m)"),

        dict(kind="section_u_z",
             lat=0, z_max=400,
             title="Mean zonal section"),

        dict(kind="section_u_z_diff",
             lat=0, z_max=400,
             title="Bias: model − reanalysis"),

        dict(kind="timeseries_euc",
             lat=0, lon=180,
             title="EUC core speed  180°E"),

        dict(kind="spectrum_u",
             lat=0, lon=180, depth=-100,
             mark_periods=[1, 3, 5],
             title="PSD  180°E, 100 m"),

        dict(kind="vertical_profile_u",
             lat=0, lon=180, z_max=400,
             title="Vertical profile  180°E"),
    ],
    ncols=2,
    figsize=(14, 16),
    suptitle="Equatorial Pacific  –  MITgcm vs Synthetic Reanalysis",
    save="panel_overview.png",
)
plt.close()
print("Saved: panel_overview.png")


# ── 4. Register a CUSTOM plot type at runtime ─────────────────────────────────
#
#  Any module can call @register_plot to add new plot types.
#  They immediately become available via dp.plot() and dp.panel().

from mitplot import register_plot

@register_plot("zonal_mean_u")
def plot_zonal_mean_u(ds, ax, *, var="UVEL", z_dim="Z", **_):
    """Latitude–depth plot of time-mean zonal-mean velocity."""
    u = ds[var].mean(["time", "XC"])
    depths = np.abs(ds[z_dim].values)
    lats   = ds["YC"].values
    cf = ax.contourf(lats, depths, u.values.T,
                     levels=20, cmap="RdBu_r", extend="both")
    ax.set_ylim(400, 0)
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Zonal-mean U")
    return cf

fig, ax = dp.plot("zonal_mean_u", figsize=(7, 4))
plt.savefig("custom_zonal_mean.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: custom_zonal_mean.png")


# ── 5. Use the low-level registry directly ────────────────────────────────────

from mitplot import get_plotter, list_plotters

print("\nAll registered plotters:")
for name in list_plotters():
    print(f"  {name}")

fn = get_plotter("timeseries_euc")
fig, ax = plt.subplots(figsize=(10, 3))
fn(ds_model, ax, lat=0, lon=180, ds_ref=ds_ref,
   model_label="MITgcm", ref_label="GLORYS12")
plt.savefig("low_level_euc.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: low_level_euc.png")
