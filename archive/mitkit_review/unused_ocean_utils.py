"""Unreferenced legacy helpers; retained for review, not installed."""
import os
import numpy as np
import xarray as xr
from scipy.signal import butter, sosfiltfilt, buttord
from mitkit.paths import project_root

def iafilt(data, fs=12, rmclim=True,axis=0,dim='time'):
    is_da = isinstance(data, xr.DataArray)
    if is_da:
        da = data
        axis = da.get_axis_num(dim)
        data = da.values
    arr = data.copy()
    # 去除月气候态，假设是连续数据
    if rmclim:
        arr = np.moveaxis(arr, axis, 0)
        for i in range(fs):
            arr[i::fs] -= arr[i::fs].mean(axis=0)
        arr = np.moveaxis(arr, 0, axis)
    # 低通：保留 2 年以上，压制 1.5 年及以下
    n, Wn = buttord(1/2, 1/1.5, 1,15, fs=fs)
    sos = butter(n, Wn, btype='low', fs=fs, output='sos')
    arr = sosfiltfilt(sos, arr, axis=axis)
    if is_da:
        return xr.DataArray(arr, coords=da.coords, dims=da.dims, attrs=da.attrs, name=da.name)
    return arr

def find_mon_file(dir_pre='mnc_exp_', base_path=None, file_pre='stat',print_path=False):
    """
    查找符合条件的文件路径。

    参数:
        base_path (str): 主目录路径。
        dir_pre (str): 子目录名称前缀，例如 'mnc_exp_'。
        file_pre (str): 文件名称前缀，例如 'stat'。

    返回:
        list: 符合条件的文件路径列表（找到第一个符合条件的文件夹后立即退出）。
    """
    if base_path is None:
        base_path = project_root() / 'run'
    matching_files = []
    dirs = []
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        # 必须是目录，且名称符合 dir_pre + 纯数字
        if os.path.isdir(full_path) and entry.startswith(dir_pre):
            suffix = entry[len(dir_pre):]
            if suffix.isdigit():
                dirs.append(entry)
    dirs.sort(reverse=True)
    # 遍历排序后的目录，找第一个包含目标文件的
    for d in dirs:
        dir_path = os.path.join(base_path, d)
        # 列出子目录中的所有文件，检查是否以指定前缀开头并且紧接着是 '.'
        matching_files = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.split('.')[0] == file_pre]
        if matching_files:
            if print_path:
                print(f"Found in directory: {dir_path}")
            break  # 找到就退出

    return matching_files
