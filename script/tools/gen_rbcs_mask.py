#!/usr/bin/env python3
"""
读取 MITgcm data 文件，生成东赤道 RBCS 波动吸收层。

输出文件：
    rbcs_mask_east_TS.bin
    rbcs_mask_east_U.bin
    rbcs_mask_east_V.bin
    rbcs_mask_east_preview.png

二进制格式：
    大端 float32，逻辑维度为 (Z, Y, X)，X 变化最快。
"""

from pathlib import Path

import f90nml
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 用户配置：只需要修改这里
# ============================================================

# MITgcm 的 data 文件
DATA_FILE = Path("/public/home/zhanghang/eqp-dlfv-mitgcm-era5/input/data")

# 输出目录；设为 None 时输出到 data 所在目录
OUTPUT_DIR = None

# 输出文件名前缀
OUTPUT_PREFIX = "rbcs_mask_east"

# 吸收层纬向范围
#
# X_START：吸收层西侧起点，此处 mask=0
# X_FULL ：完整吸收开始位置，此处及其以东 mask=1
#
# 可以使用 -180~180 或 0~360 经度：
# -100° = 100°W = 260°E
#  -95° =  95°W = 265°E
X_START = -105.0
X_FULL = -100.0

# 吸收层经向范围
#
# |lat| <= Y_CORE：mask_y=1
# Y_CORE < |lat| < Y_EDGE：平滑减小
# |lat| >= Y_EDGE：mask_y=0
Y_CORE = 5.5
Y_EDGE = 7.5

# 是否生成预览图
SAVE_FIGURE = True


# ============================================================
# 读取 MITgcm 网格
# ============================================================

def as_1d_array(value, name):
    """将 f90nml 读取结果转为完整的一维浮点数组。"""
    if value is None:
        raise KeyError(f"data 中没有找到 {name}")

    raw = np.asarray(value, dtype=object).ravel()

    if raw.size == 0:
        raise ValueError(f"{name} 为空")

    if any(item is None for item in raw):
        raise ValueError(f"{name} 中存在未赋值元素")

    array = raw.astype(np.float64)

    if array.size == 1:
        raise ValueError(
            f"{name} 只有一个数值，无法仅凭 data 确定网格尺寸。\n"
            f"请使用类似 {name} = 720*0.25, 的完整写法。"
        )

    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} 包含 NaN 或无穷值")

    if np.any(array <= 0):
        raise ValueError(f"{name} 必须全部大于 0")

    return array


def read_grid_from_data(data_file):
    """
    从 MITgcm data 的 PARM04 中读取：

        delX, delY, delR
        xgOrigin, ygOrigin

    并计算：

        XC, YC：C 点坐标
        XG, YG：U/V 点坐标
        RC, RF：垂向中心和界面
    """
    if not data_file.is_file():
        raise FileNotFoundError(f"找不到 data 文件：{data_file}")

    nml = f90nml.read(data_file)
    parm04 = nml["parm04"]

    delx = as_1d_array(parm04.get("delx"), "delX")
    dely = as_1d_array(parm04.get("dely"), "delY")
    delr = as_1d_array(parm04.get("delr"), "delR")

    xg_origin = float(parm04.get("xgorigin", 0.0))
    yg_origin = float(parm04.get("ygorigin", 0.0))

    nx = delx.size
    ny = dely.size
    nr = delr.size

    # 各网格单元西侧面坐标
    xg = xg_origin + np.r_[0.0, np.cumsum(delx[:-1])]

    # 各网格单元南侧面坐标
    yg = yg_origin + np.r_[0.0, np.cumsum(dely[:-1])]

    # C 点中心坐标
    xc = xg + 0.5 * delx
    yc = yg + 0.5 * dely

    # 垂向坐标，向下为负
    rf = -np.r_[0.0, np.cumsum(delr)]
    rc = 0.5 * (rf[:-1] + rf[1:])

    return {
        "nx": nx,
        "ny": ny,
        "nr": nr,
        "delX": delx,
        "delY": dely,
        "delR": delr,
        "XC": xc,
        "YC": yc,
        "XG": xg,
        "YG": yg,
        "RC": rc,
        "RF": rf,
    }


# ============================================================
# 经度转换
# ============================================================

def align_longitude(longitude, model_longitude):
    """
    将输入经度调整到模型经度范围附近。

    例如模型范围为 112~292°E：
        -100° 自动转换为 260°E。
    """
    model_center = 0.5 * (
        model_longitude.min() + model_longitude.max()
    )

    candidates = longitude + 360.0 * np.arange(-3, 4)

    return float(
        candidates[
            np.argmin(np.abs(candidates - model_center))
        ]
    )


def resolve_longitude_range(x_start, x_full, model_longitude):
    """转换并检查吸收层的纬向范围。"""
    start = align_longitude(x_start, model_longitude)
    full = align_longitude(x_full, model_longitude)

    while full <= start:
        full += 360.0

    xmin = float(model_longitude.min())
    xmax = float(model_longitude.max())

    if not xmin <= start <= xmax:
        raise ValueError(
            f"X_START={start:.3f}° 不在模型范围 "
            f"{xmin:.3f}°~{xmax:.3f}° 内"
        )

    if not start < full <= xmax:
        raise ValueError(
            f"X_FULL={full:.3f}° 不在模型范围内，"
            f"或没有位于 X_START 以东"
        )

    return start, full


# ============================================================
# 生成吸收层
# ============================================================

def half_cosine_increase(x, start, end):
    """
    半余弦递增：

        x <= start：0
        start < x < end：从0平滑增加到1
        x >= end：1
    """
    if end <= start:
        raise ValueError("end 必须大于 start")

    phase = np.clip(
        (x - start) / (end - start),
        0.0,
        1.0,
    )

    return 0.5 * (1.0 - np.cos(np.pi * phase))


def equatorial_taper(latitude, core, edge):
    """
    赤道带经向衰减：

        |lat| <= core：1
        core < |lat| < edge：由1平滑减小到0
        |lat| >= edge：0
    """
    if core < 0:
        raise ValueError("Y_CORE 不能小于 0")

    if edge <= core:
        raise ValueError("Y_EDGE 必须大于 Y_CORE")

    phase = np.clip(
        (np.abs(latitude) - core) / (edge - core),
        0.0,
        1.0,
    )

    return 0.5 * (1.0 + np.cos(np.pi * phase))


def make_absorbing_mask(
    longitude,
    latitude,
    x_start,
    x_full,
    y_core,
    y_edge,
):
    """
    生成二维东赤道吸收层。

    返回形状：
        (ny, nx)
    """
    mask_x = half_cosine_increase(
        longitude,
        x_start,
        x_full,
    )

    mask_y = equatorial_taper(
        latitude,
        y_core,
        y_edge,
    )

    mask = mask_y[:, None] * mask_x[None, :]

    return mask.astype(np.float32)


# ============================================================
# 二进制输出
# ============================================================

def write_3d_mask(mask_2d, nr, output_file):
    """
    将二维 mask 在垂向重复 nr 次，写成 MITgcm 裸二进制文件。

    不创建完整三维数组，避免不必要的内存占用。
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plane = np.ascontiguousarray(mask_2d, dtype=">f4")

    with output_file.open("wb") as file:
        for _ in range(nr):
            plane.tofile(file)

    expected_size = nr * mask_2d.size * 4
    actual_size = output_file.stat().st_size

    if actual_size != expected_size:
        raise IOError(
            f"{output_file} 文件大小错误："
            f"{actual_size} != {expected_size}"
        )


def save_preview(
    output_file,
    longitude,
    latitude,
    mask,
    x_start,
    x_full,
):
    """保存 T/S 掩膜预览图。"""
    fig, ax = plt.subplots(
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    mesh = ax.pcolormesh(
        longitude,
        latitude,
        mask,
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.axvline(
        x_start,
        linestyle="--",
        label=f"start: {x_start:.2f}°",
    )
    ax.axvline(
        x_full,
        linestyle="--",
        label=f"full: {x_full:.2f}°",
    )

    ax.axhline(Y_CORE, linestyle=":")
    ax.axhline(-Y_CORE, linestyle=":")
    ax.axhline(Y_EDGE, linestyle=":")
    ax.axhline(-Y_EDGE, linestyle=":")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Eastern Equatorial RBCS Absorbing Mask")
    ax.legend()

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("RBCS mask")

    fig.savefig(output_file, dpi=200)
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================

def main():
    data_file = DATA_FILE.expanduser().resolve()

    if OUTPUT_DIR is None:
        output_dir = data_file.parent
    else:
        output_dir = Path(OUTPUT_DIR).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取网格
    grid = read_grid_from_data(data_file)

    # 经度转换和检查
    x_start, x_full = resolve_longitude_range(
        X_START,
        X_FULL,
        grid["XC"],
    )

    # T/S：C 点
    mask_ts = make_absorbing_mask(
        longitude=grid["XC"],
        latitude=grid["YC"],
        x_start=x_start,
        x_full=x_full,
        y_core=Y_CORE,
        y_edge=Y_EDGE,
    )

    # U：西侧网格面，坐标为 XG、YC
    mask_u = make_absorbing_mask(
        longitude=grid["XG"],
        latitude=grid["YC"],
        x_start=x_start,
        x_full=x_full,
        y_core=Y_CORE,
        y_edge=Y_EDGE,
    )

    # V：南侧网格面，坐标为 XC、YG
    mask_v = make_absorbing_mask(
        longitude=grid["XC"],
        latitude=grid["YG"],
        x_start=x_start,
        x_full=x_full,
        y_core=Y_CORE,
        y_edge=Y_EDGE,
    )

    output_ts = output_dir / f"{OUTPUT_PREFIX}_TS.bin"
    output_u = output_dir / f"{OUTPUT_PREFIX}_U.bin"
    output_v = output_dir / f"{OUTPUT_PREFIX}_V.bin"
    # output_npz = Path('.') / f"{OUTPUT_PREFIX}_preview.npz"
    output_png = Path('.') / f"{OUTPUT_PREFIX}_preview.png"

    # 输出三维 mask
    write_3d_mask(mask_ts, grid["nr"], output_ts)
    write_3d_mask(mask_u, grid["nr"], output_u)
    write_3d_mask(mask_v, grid["nr"], output_v)

    # 保存二维数据，便于检查
    # np.savez_compressed(
    #     output_npz,
    #     XC=grid["XC"],
    #     YC=grid["YC"],
    #     XG=grid["XG"],
    #     YG=grid["YG"],
    #     mask_TS=mask_ts,
    #     mask_U=mask_u,
    #     mask_V=mask_v,
    #     x_start=x_start,
    #     x_full=x_full,
    #     y_core=Y_CORE,
    #     y_edge=Y_EDGE,
    # )

    if SAVE_FIGURE:
        save_preview(
            output_png,
            grid["XC"],
            grid["YC"],
            mask_ts,
            x_start,
            x_full,
        )

    print()
    print("RBCS 吸收层生成完成")
    print("=" * 60)
    print(f"data             : {data_file}")
    print(
        f"grid             : "
        f"nr={grid['nr']}, ny={grid['ny']}, nx={grid['nx']}"
    )
    print(
        f"longitude        : "
        f"{grid['XC'].min():.3f} ~ {grid['XC'].max():.3f}"
    )
    print(
        f"latitude         : "
        f"{grid['YC'].min():.3f} ~ {grid['YC'].max():.3f}"
    )
    print(f"zonal ramp       : {x_start:.3f} ~ {x_full:.3f}")
    print(f"latitude core    : ±{Y_CORE:.3f}")
    print(f"latitude edge    : ±{Y_EDGE:.3f}")
    print(f"mask range       : {mask_ts.min():.6f} ~ {mask_ts.max():.6f}")
    print(f"T/S mask         : {output_ts}")
    print(f"U mask           : {output_u}")
    print(f"V mask           : {output_v}")
    # print(f"preview data     : {output_npz}")

    if SAVE_FIGURE:
        print(f"preview figure   : {output_png}")

    print("=" * 60)


if __name__ == "__main__":
    main()