import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import butter, sosfiltfilt, hilbert
from matplotlib.ticker import FuncFormatter

work_dir = Path(os.environ['WORK_DIR'])
sys.path.append(str(work_dir/'script'))
from mit_utils import open_mds

# ========================= 参数 =========================
exp = "260728_165832_rb1"
PATH = work_dir/"output"/exp
VAR = "UVEL"

TIME_RANGE = None
# TIME_RANGE = ("1991-01", "2025-12")

FREQ_RANGE = (0.55, 0.85)      # cycle/year，对应1–2年
FS = 12                      # 月平均数据
FILTER_ORDER = 4
NUM_MODES = 6
MODE = 1

DEEP_LON = (114, 280)
DEEP_LAT = (-0.5, 0.5)
DEEP_DEPTH = (-200, -4000)

UPPER_LON = (114, 290)
UPPER_LAT = (-10, 10)
UPPER_DEPTH = (0, -200)

MAP_COARSEN = 1              # 高分辨率模式可设为4或6，1表示不粗化
PHASE_MASK = 0.10            # 掩膜振幅低于最大值10%的相位，设为0则关闭


# ========================= 数据读取 =========================

ds = open_mds(PATH,prefix='diag3d')
u = ds[VAR]
print(u)
rename = {
    "XG": "lon", "YC": "lat", "Z": "depth"
}
u = u.rename({k: v for k, v in rename.items() if k in u.dims or k in u.coords})

if TIME_RANGE is not None:
    u = u.sel(time=slice(*TIME_RANGE))

# 若模式深度坐标向下为负，取消下一行注释
# u = u.assign_coords(depth=np.abs(u.depth)).sortby("depth")


# ========================= CEOF =========================

def bandpass(X, freq=FREQ_RANGE, fs=FS, order=FILTER_ORDER):
    """Butterworth零相位带通，X维度为(time, space)。"""
    X = X.copy()
    for i in range(fs):
        X[i::fs] -= X[i::fs].mean(axis=0, keepdims=True)
    sos = butter(order, freq, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=0)


def ceof(da, num_modes=NUM_MODES):
    """适用于(time, depth, lon)断面或(time, lat, lon)大面。"""
    spatial_dims = [d for d in da.dims if d != "time"]
    da = da.transpose("time", *spatial_dims)

    nt = da.sizes["time"]
    spatial_shape = da.shape[1:]
    X = np.asarray(da.values, dtype=float).reshape(nt, -1)

    valid = np.all(np.isfinite(X), axis=0)
    valid[valid] &= X[:, valid].std(axis=0) > 0

    if not valid.any():
        raise ValueError("没有完整且具有时间变化的有效格点。")

    Xa = hilbert(bandpass(X[:, valid]), axis=0)

    eigenvalues, eigenvectors = np.linalg.eigh(Xa @ Xa.conj().T)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[order].real, 0)

    nmode = min(num_modes, nt)
    eigenvectors = eigenvectors[:, order[:nmode]]

    tc = eigenvectors * np.sqrt(nt)
    sm_valid = eigenvectors.conj().T @ Xa / np.sqrt(nt)

    # 以第一个分析时刻为相位基准：tc(t0)为正实数
    rotation = np.exp(-1j * np.angle(tc[0]))
    tc *= rotation
    sm_valid *= np.conj(rotation)[:, None]

    sm = np.full((nmode, X.shape[1]), np.nan + 1j * np.nan, dtype=complex)
    sm[:, valid] = sm_valid
    sm = sm.reshape(nmode, *spatial_shape)

    modes = np.arange(1, nmode + 1)
    spatial_coords = {"mode": modes, **{d: da[d] for d in spatial_dims}}

    return xr.Dataset({
        "spatial_mode": xr.DataArray(sm, dims=("mode", *spatial_dims), coords=spatial_coords),
        "time_coefficient": xr.DataArray(tc, dims=("time", "mode"), coords={"time": da.time, "mode": modes}),
        "explained_variance": xr.DataArray(eigenvalues[:nmode] / eigenvalues.sum(), dims="mode", coords={"mode": modes})
    })


# ========================= 数据预处理 =========================

deep = u.sel(lon=slice(*DEEP_LON), lat=slice(*DEEP_LAT), depth=slice(*DEEP_DEPTH)).mean("lat", skipna=True)
deep = deep.transpose("time", "depth", "lon")

upper = u.sel(lon=slice(*UPPER_LON), lat=slice(*UPPER_LAT), depth=slice(*UPPER_DEPTH)).mean("depth", skipna=True)

if MAP_COARSEN > 1:
    upper = upper.coarsen(lat=MAP_COARSEN, lon=MAP_COARSEN, boundary="trim").mean()

upper = upper.transpose("time", "lat", "lon")


# ========================= 执行CEOF =========================

deep_result = ceof(deep)
upper_result = ceof(upper)

print("Deep-section explained variance (%):")
print(np.round(deep_result.explained_variance.values * 100, 2))

print("\nUpper-layer explained variance (%):")
print(np.round(upper_result.explained_variance.values * 100, 2))


# ========================= 绘图工具 =========================

def lon_formatter(x, pos=None):
    x = x % 360
    return f"{360 - x:g}°W" if x > 180 else f"{x:g}°E"


def lat_formatter(y, pos=None):
    if y < 0:
        return f"{abs(y):g}°S"
    if y > 0:
        return f"{y:g}°N"
    return "0°"


def get_phase(sm):
    """保持原代码的-angle相位定义，并掩膜低振幅区域。"""
    amplitude = np.abs(sm)
    phase = -np.angle(sm)

    if PHASE_MASK > 0:
        phase = np.where(amplitude >= PHASE_MASK * amplitude.max(), phase, np.nan)

    return phase


def plot_time_coefficient(ax, tc, variance, mode):
    ax.plot(tc.time, tc.real, label="Real")
    ax.plot(tc.time, tc.imag, label="Imaginary")
    ax.plot(tc.time, np.abs(tc), "k:", label="Magnitude")
    ax.plot(tc.time, -np.abs(tc), "k:")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title(f"(c) CEOF{mode} Time Coefficients, $R^2={variance * 100:.1f}\\%$")
    ax.set_xlabel("Time")
    ax.legend(loc="upper center", ncol=3, frameon=False)


# ========================= 深层断面绘图 =========================

def plot_deep(result, mode=MODE):
    sm = result.spatial_mode.sel(mode=mode)
    tc = result.time_coefficient.sel(mode=mode)
    sm['depth'] = -sm.depth
    variance = float(result.explained_variance.sel(mode=mode))

    amplitude = np.abs(sm) * 100
    phase = get_phase(sm)
    levels = np.linspace(0, float(amplitude.quantile(0.99)), 11)

    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.5], wspace=0.1, hspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    p1 = ax1.contourf(sm.lon, sm.depth, amplitude, levels=levels, cmap="RdYlBu_r", extend="max")
    ax1.set_ylim(float(sm.depth.max()), float(sm.depth.min()))
    ax1.set_title(f"(a) Magnitude CEOF{mode}")
    ax1.set_ylabel("Depth (m)")
    ax1.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    fig.colorbar(p1, ax=ax1, orientation="horizontal", pad=0.12, aspect=25, label="cm s$^{-1}$")

    ax2 = fig.add_subplot(gs[0, 1])
    p2 = ax2.pcolormesh(sm.lon, sm.depth, phase, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, shading="auto")
    ax2.set_ylim(float(sm.depth.max()), float(sm.depth.min()))
    ax2.set_title(f"(b) Phase CEOF{mode}")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Depth (m)")
    ax2.xaxis.set_major_formatter(FuncFormatter(lon_formatter))

    cbar = fig.colorbar(p2, ax=ax2, orientation="horizontal", pad=0.12, aspect=25)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    ax3 = fig.add_subplot(gs[1, :])
    plot_time_coefficient(ax3, tc, variance, mode)

    return fig


# ========================= 上层大面绘图 =========================

def plot_upper(result, mode=MODE):
    sm = result.spatial_mode.sel(mode=mode)
    tc = result.time_coefficient.sel(mode=mode)
    variance = float(result.explained_variance.sel(mode=mode))

    amplitude = np.abs(sm) * 100
    phase = get_phase(sm)
    levels = np.linspace(0, float(amplitude.quantile(0.99)), 11)

    fig = plt.figure(figsize=(11, 6.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.5], wspace=0.12, hspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    p1 = ax1.contourf(sm.lon, sm.lat, amplitude, levels=levels, cmap="RdYlBu_r", extend="max")
    ax1.set_title(f"(a) Magnitude CEOF{mode}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax1.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    fig.colorbar(p1, ax=ax1, orientation="horizontal", pad=0.12, aspect=25, label="cm s$^{-1}$")

    ax2 = fig.add_subplot(gs[0, 1])
    p2 = ax2.pcolormesh(sm.lon, sm.lat, phase, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, shading="auto")
    ax2.set_title(f"(b) Phase CEOF{mode}")
    ax2.set_xlabel("Longitude")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Latitude")
    ax2.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

    cbar = fig.colorbar(p2, ax=ax2, orientation="horizontal", pad=0.12, aspect=25)
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    ax3 = fig.add_subplot(gs[1, :])
    plot_time_coefficient(ax3, tc, variance, mode)

    return fig


# ========================= 绘制第一模态 =========================

fig_deep = plot_deep(deep_result, mode=1)
plt.savefig(f"{exp}_CEOF_deep.png", bbox_inches="tight", dpi=300)

fig_upper = plot_upper(upper_result, mode=1)
plt.savefig(f"{exp}_CEOF_upper.png", bbox_inches="tight", dpi=300)