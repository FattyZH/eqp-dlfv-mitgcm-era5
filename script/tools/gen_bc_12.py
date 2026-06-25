from pathlib import Path
import f90nml
import numpy as np
import xarray as xr
from scipy.ndimage import label, distance_transform_edt, binary_dilation


# =========================
# 0. 基本配置
# =========================

X_RANGE_DEG = 180
Y_RANGE_DEG = 48
RES = 12

NX = int(X_RANGE_DEG * RES)
NY = int(Y_RANGE_DEG * RES)

X0 = 112
Y0 = -23.5
X1 = X0 + (NX - 1) / RES
Y1 = Y0 + (NY - 1) / RES

DELTA_T = 450.0
MINIMUM_DEPTH = 7.2
SPONGE_THICKNESS = 25

IPATH = Path.home() / "data"
BASEDIR = Path.home() / "eqp-dlfv-mitgcm-era5"
OPATH = BASEDIR / "input"

FBATH = IPATH / "GLO-MFC_001_030_mask_bathy.nc"
FCOOR = IPATH / "GLO-MFC_001_030_coordinates.nc"
FCLIM = IPATH / "cmems_climatology_mon.nc"

DATA_NML_IN = BASEDIR / "config/data"
DATA_NML_OUT = BASEDIR / "config/data.gen"

OBCS_NML_IN = BASEDIR / "config/data.obcs"
OBCS_NML_OUT = BASEDIR / "config/data.obcs.gen"


# =========================
# 1. 工具函数
# =========================

def write_be_f4(arr, path):
    """以 MITgcm 常用 big-endian float32 写出二进制文件。"""
    arr.astype(">f4").tofile(path)


def fill_nearest_nan(arr):
    """
    使用最近邻非 NaN 值填充 NaN。
    与原 fillna(arr) 逻辑一致。
    """
    filled = arr.copy()

    if np.isnan(arr).any():
        nan_mask = np.isnan(arr)
        idx = distance_transform_edt(
            nan_mask,
            return_distances=False,
            return_indices=True,
        )
        filled[:] = arr[tuple(idx)]

    return filled


def print_grid_info():
    print("Nx,Ny=", NX, NY)
    print("x1,y1=", X1, Y1)


def print_vertical_grid(dR):
    print(f"Nr  = {len(dR):4d}")
    print("H   =", dR.sum())

    print(" delR=", end="")
    for i, x in enumerate(dR):
        if i % 10 == 0 and i > 0:
            print()
            print("      ", end="")
        print(f"{x:6.2f},", end="")
    print()


# =========================
# 2. 水平与垂向网格
# =========================

def build_vertical_grid(fcoor):
    """
    从 GLORYS 坐标文件中读取 e3t，并合并垂向层。
    保持原始 Rid 逻辑不变。
    """
    with xr.open_dataset(fcoor) as ds:
        dr = ds["e3t"].values

    Rid = [0, 3, 6, 8, 10] + [i for i in range(12, dr.size)]

    dR = np.zeros(len(Rid) - 1)
    for k in range(len(Rid) - 1):
        dR[k] = dr[Rid[k]:Rid[k + 1]].sum()

    return dr, Rid, dR


def update_data_namelist(nml, dR):
    """更新 data namelist 中与网格和时间步长相关的参数。"""
    nml.repeat_counter = True

    nml["PARM03"]["deltaT"] = DELTA_T
    nml["PARM03"]["dumpFreq"] = 0.0

    nml["PARM04"]["delX"] = [1 / RES] * NX
    nml["PARM04"]["delY"] = [1 / RES] * NY
    nml["PARM04"]["xgOrigin"] = X0 - 1 / 2  / RES
    nml["PARM04"]["ygOrigin"] = Y0 - 1 / 2  / RES
    nml["PARM04"]["delR"] = dR

    return nml


# =========================
# 3. 地形
# =========================

def read_domain_bathymetry(fbath):
    """
    读取并拼接跨 180° 经度的数据。
    返回目标区域二维地形 DataArray。
    """
    with xr.open_dataset(fbath) as ds:
        da = ds["deptho"]

        hl = da.loc[Y0:Y1, X0:X1]
        hr = da.loc[Y0:Y1, X0 - 360:X1 - 360]
        hr["longitude"] = hr["longitude"] + 360

    return xr.concat([hl, hr], dim="longitude")


def keep_largest_connected_ocean(depth, ignore_outer=False):
    """
    保留最大连通湿区。
    返回：
    - depth_arr: 处理后的地形数组，陆地为 0
    - land_mask: True 表示陆地或非最大连通湿区
    """
    if hasattr(depth, "values"):
        depth_arr = depth.values.copy()
    else:
        depth_arr = depth.copy()
    if depth_arr.shape != (NY, NX):
        raise ValueError(f"Expected shape {(NY, NX)}, but got {depth_arr.shape}")
    wet = np.isfinite(depth_arr) & (depth_arr > 0)
    wet_for_label = wet.copy()
    if ignore_outer:
        outer_mask = np.zeros_like(wet, dtype=bool)
        outer_mask[0, :] = True
        outer_mask[-1, :] = True
        outer_mask[:, 0] = True
        outer_mask[:, -1] = True
        # 连通检测时，四周最外层临时视为陆地
        wet_for_label[outer_mask] = False

    labels, _ = label(wet_for_label)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    land_mask = labels != sizes.argmax()
    if ignore_outer:
        adjacent_to_main = binary_dilation(~land_mask)

        keep = (~land_mask) | (wet & outer_mask & adjacent_to_main)
        land_mask = ~keep

    depth_arr[land_mask] = np.nan

    return depth_arr, land_mask


def generate_bathymetry(nml):
    """生成 MITgcm 地形文件，并更新 namelist。"""
    depth = read_domain_bathymetry(FBATH)
    depth, land_mask0 = keep_largest_connected_ocean(depth)
    depth, land_mask = keep_largest_connected_ocean(depth, ignore_outer=True)
    print(
        f"Removed {np.count_nonzero(land_mask & (~land_mask0))} wet points "
        "connected only through outer boundary layer"
    )
    depth[depth < MINIMUM_DEPTH] = MINIMUM_DEPTH
    depth[np.isnan(depth)] = 0
    bathy = -depth
    relpath = "bathy12.bin"
    write_be_f4(bathy, OPATH / relpath)

    print(f"Written bathymetry to {relpath}")

    nml["PARM05"]["bathyFile"] = relpath

    return nml, land_mask


# =========================
# 4. 初始场
# =========================

def vertical_average_3d(data, dr, Rid, dR):
    """
    将 GLORYS 原始垂向层加权平均到 MITgcm 垂向层。
    输入 data shape: (z, y, x)
    输出 datai shape: (nz, y, x)
    """
    nz = len(dR)
    out = np.zeros((nz, NY, NX))

    for k in range(nz):
        zslice = slice(Rid[k], Rid[k + 1])
        out[k] = (data[zslice] * dr[zslice, None, None]).sum(0) / dR[k]

    return out


def prepare_initial_field(ds, var, land_mask, dr, Rid, dR):
    """
    生成单个变量的初始场。
    原逻辑：
    - 12月和1月平均；
    - zos 为二维；
    - uo/vo/thetao/so 为三维，并做垂向加权平均；
    - 陆地或非最大连通湿区用最近邻填充。
    """
    data = ds[var][[0, 11]].mean("time")
    data = data.sel(latitude=slice(Y0, Y1), longitude=slice(X0, X1))
    data = data.values

    if var == "zos":
        data[land_mask] = np.nan
        return fill_nearest_nan(data)

    data[:, land_mask] = np.nan

    for k in range(data.shape[0]):
        data[k] = fill_nearest_nan(data[k])

    return vertical_average_3d(data, dr, Rid, dR)


def generate_initial_conditions(nml, land_mask, dr, Rid, dR):
    """生成初始场文件，并更新 data namelist。"""
    var_names = ["uo", "vo", "thetao", "so", "zos"]

    nml_names = [
        "uVelInitFile",
        "vVelInitFile",
        "hydrogThetaFile",
        "hydrogSaltFile",
        "pSurfInitFile",
    ]

    with xr.open_dataset(FCLIM) as ds:
        for var, nml_name in zip(var_names, nml_names):
            field = prepare_initial_field(ds, var, land_mask, dr, Rid, dR)

            print(f"{var} shape: {field.shape}")

            relpath = f"init/{var}_12.bin"
            write_be_f4(field, OPATH / relpath)

            nml["PARM05"][nml_name] = relpath

    return nml


# =========================
# 5. 侧边界气候态
# =========================

def vertical_average_obc(arr, dr, Rid, dR):
    """
    对 OBCS 边界场做垂向加权平均。

    输入 arr shape:
    - (time, z, x) for S/N
    - (time, z, y) for W/E

    输出 shape:
    - (time, nz, boundary_points)
    """
    nt = arr.shape[0]
    nz = len(dR)
    nb = arr.shape[2]

    out = np.zeros((nt, nz, nb))

    for k in range(nz):
        zslice = slice(Rid[k], Rid[k + 1])
        out[:, k] = (arr[:, zslice] * dr[zslice, None]).sum(1) / dR[k]

    return out


def prepare_obc_field(ds, var, obc, land_mask, dr, Rid, dR):
    """
    生成某个 OBCS 边界上的某个变量。
    保持原始 loc 选择方式和 mask 方式。
    """
    xso_map = {"W": X0, "E": X1}
    yso_map = {"S": Y0, "N": Y1}

    obc_land_mask = {
        "S": land_mask[0, :],
        "N": land_mask[-1, :],
        "W": land_mask[:, 0],
        "E": land_mask[:, -1],
    }

    xso = xso_map.get(obc, slice(X0, X1))
    yso = yso_map.get(obc, slice(Y0, Y1))

    arr = ds[var].loc[:, :, yso, xso].values

    arr[:, :, obc_land_mask[obc]] = np.nan

    for it in range(arr.shape[0]):
        for iz in range(arr.shape[1]):
            arr[it, iz] = fill_nearest_nan(arr[it, iz])

    return vertical_average_obc(arr, dr, Rid, dR)


def generate_obcs(land_mask, dr, Rid, dR):
    """生成 OBCS 文件，并写出 data.obcs.gen。"""
    obcs = ["S", "N", "W"]
    var_names = ["thetao", "so", "uo", "vo"]

    obnml = f90nml.read(OBCS_NML_IN)
    obnml["OBCS_PARM03"]["spongeThickness"] = SPONGE_THICKNESS

    nz = len(dR)

    with xr.open_dataset(FCLIM) as ds:
        for obc in obcs:
            print(f"处理{obc}边界")

            for var in var_names:
                field = prepare_obc_field(ds, var, obc, land_mask, dr, Rid, dR)

                relpath = f"obcs/OB{obc}{var[0]}_12_{nz}.bin"
                print(f"{relpath} shape:", field.shape)

                write_be_f4(field, OPATH / relpath)

                obnml["OBCS_PARM01"][f"OB{obc}{var[0]}File"] = relpath

    obnml.write(OBCS_NML_OUT, force=True)


# =========================
# 6. 主流程
# =========================

def main():
    print_grid_info()

    dr, Rid, dR = build_vertical_grid(FCOOR)
    print_vertical_grid(dR)

    nml = f90nml.read(DATA_NML_IN)
    nml = update_data_namelist(nml, dR)

    nml, land_mask = generate_bathymetry(nml)
    nml = generate_initial_conditions(nml, land_mask, dr, Rid, dR)

    nml.write(DATA_NML_OUT, force=True)

    generate_obcs(land_mask, dr, Rid, dR)


if __name__ == "__main__":
    main()