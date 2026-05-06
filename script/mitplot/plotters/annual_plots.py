import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import scipy.io as sio
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from pathlib import Path
from mitplot.registry import register_plot

def _require_var(ds: xr.Dataset, var: str) -> xr.DataArray:
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset. "
                       f"Available: {list(ds.data_vars)}")
    return ds[var]

def hmnc(z, p):
    n = z.shape[0]
    e1 = np.exp(-2j*np.pi/p*np.arange(n))
    e2 = np.exp(2j*np.pi/p*np.arange(n))
    mf1 = np.tensordot(z, e1, (0, 0))/n
    mf2 = np.tensordot(z, e2, (0, 0))/n
    r = (abs(mf1)**2+abs(mf2)**2)/np.var(z, 0)
    major = abs(mf1)+abs(mf2)
    minor = abs(mf1)-abs(mf2)
    inc = np.angle(mf1*mf2)/2
    pha = inc-np.angle(mf1)
    return r, pha, major, minor, inc

@register_plot("u_sec_eq_ann")
def plot_ann_sec(ds: xr.Dataset,
                     ax: plt.Axes | None = None,
                     lon_lim: slice = slice(130,275),
                     lat: float = 0.0,
                     time: slice | None = None,
                     z_max: float = 4400.0,
                     ):
    """Depth-longitude section of zonal velocity annual varibility."""
    if not time: 
        time = slice('2013-01-01','2024-12-31')
    da = _require_var(ds,'UVEL')
    da = da.loc[time, :, lat, lon_lim]
    time = da['time'].values.astype('M8[D]')
    lon = da['XG'].values
    lat = da['YC'].values
    dep = -da['Z'].values
    u = da.values
    u[:,np.all(u == 0, axis=0)] = np.nan
    r1, ph1,maj1 = hmnc(u, 12)[:3]
    mph1 = (12/(2*np.pi)*ph1+.5) % 12+1
    tray = sio.loadmat(Path(__file__).parent/'rossbytray.mat', squeeze_me=True)
    tray['x'] = tray['x']-2
    plt.rcdefaults()
    plt.rcParams.update(
        {
            'axes.facecolor': 'gray',
            # 'font.family': ['Times New Romans', 'Microsoft Yahei'],
            'image.cmap': 'RdYlBu_r',
            'contour.negative_linestyle': '-',
            'contour.linewidth': 0.8,
            'font.size': 15.0,
        })

    btxdt = {'fontsize': 16, 'weight': 'bold'}
    cbdt = dict(orientation='horizontal', aspect=25,pad=0.15)

    xl = (lon_lim.start,lon_lim.stop)
    if not ax:
        ax = plt.subplot()
    fig, spec = ax.figure, ax.get_subplotspec()
    ax.remove()
    gs = spec.subgridspec(1,2)
    ax = fig.add_subplot(gs[0])
    im = ax.contourf(lon, dep, maj1,np.arange(0, 0.21, 0.02),extend='max')
    plt.colorbar(im, **cbdt)
    ax.axvline(142, linestyle='--', color='w')
    ax.plot(tray['x'].T, tray['z'].T, 'r--')
    ax.set_xlim(xl)
    ax.set_ylim(0,z_max)
    ax.invert_yaxis()
    ax.set_title(r'Amplitude of U (m/s)', fontdict=btxdt)
    ax.set_ylabel('Depth (m)',fontsize=16)
    

    ax = fig.add_subplot(gs[1])
    im = plt.pcolormesh(lon, dep, mph1, cmap='Paired',
                vmin=1, vmax=13, shading='nearest')
    label = ['Feb', 'May', 'Aug', 'Nov']
    cb = plt.colorbar(im, **cbdt)
    cb.set_ticks(np.arange(2.5, 12, 3))
    cb.set_ticklabels(label)
    ax.axvline(142, linestyle='--', color='w')
    ax.plot(tray['x'].T, tray['z'].T, 'r--')
    ax.set_xlim(xl)
    ax.set_ylim(0,z_max)
    ax.invert_yaxis()

    ax.set_title('Phase of U (Month)', fontdict=btxdt)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    ax.set_ylabel('Depth (m)',fontsize=16)

@register_plot("u_ann_field")
def plot_ann_hor(ds: xr.Dataset,
                     ax: plt.Axes | None = None,
                     lon_lim: slice = slice(None),
                     lat_lim: float = slice(None),
                     dep: float = 0.0,
                     time: slice | None = None,
                     clevs=None,
                     ):
    """latitude-longitude field of zonal velocity annual varibility."""
    if not time: 
        time = slice('2013-01-01','2024-12-31')
    da = _require_var(ds,'UVEL')
    da = da.sel(Z=-dep,method='nearest')
    da = da.loc[time, lat_lim, lon_lim]
    time = da['time'].values.astype('M8[D]')
    lon = da['XG'].values
    lat = da['YC'].values
    u = da.values

    u[:,np.all(u == 0, axis=0)] = np.nan
    ph1,maj1 = hmnc(u, 12)[1:3]

    mph1 = (12/(2*np.pi)*ph1+.5) % 12+1
    plt.rcdefaults()
    plt.rcParams.update(
        {
            'axes.facecolor': 'gray',
            'image.cmap': 'RdYlBu_r',
            'font.size': 15.0,
        })

    btxdt = {'fontsize': 16, 'weight': 'bold'}
    cbdt = dict(orientation='horizontal', aspect=25,pad=0.15)

    xl = (lon_lim.start,lon_lim.stop)
    yl = (lat_lim.start,lat_lim.stop)

    if not ax:
        ax = plt.subplot()
    fig, spec = ax.figure, ax.get_subplotspec()
    ax.remove()
    gs = spec.subgridspec(1,2)
    ax = fig.add_subplot(gs[0])
    im = ax.contourf(lon, lat, maj1,np.arange(0, 0.21, 0.02),extend='max')
    plt.colorbar(im, **cbdt)
    ax.axhline(0, linestyle='--', color='k', alpha=0.7)
    ax.axvline(142, linestyle='--', color='w')
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_title('Amplitude of U (m/s)', fontdict=btxdt)
    

    ax = fig.add_subplot(gs[1])
    im = plt.pcolormesh(lon, lat, mph1, cmap='Paired',
                vmin=1, vmax=13, shading='nearest')
    label = ['Feb', 'May', 'Aug', 'Nov']
    cb = plt.colorbar(im, **cbdt)
    cb.set_ticks(np.arange(2.5, 12, 3))
    cb.set_ticklabels(label)
    ax.axhline(0, linestyle='--', color='k', alpha=0.7)
    ax.axvline(142, linestyle='--', color='w')
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_title('Phase of U (Month)', fontdict=btxdt)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
