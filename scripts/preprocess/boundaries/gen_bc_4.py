#!/usr/bin/env python3
"""Generate quarter-degree GL12 initial fields and monthly OBCS only.

Uses the agreed original layer groups with e3t thicknesses, a common largest
surface-basin mask, horizontal smoothing and per-native-layer nearest filling.
Writes only to an independent output directory; never edits input/ or code/.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, distance_transform_edt, label

# ==================== 用户配置 ====================
ROOT = Path(__file__).resolve().parents[3]
CLIMATOLOGY = Path.home()/'data/cmems_climatology_mon.nc'
COORDINATES = Path.home()/'data/GLO-MFC_001_030_coordinates.nc'
SOURCE_BATHYMETRY = Path.home()/'data/GLO-MFC_001_030_mask_bathy.nc'
# 模型格点中心；直接设置，不读取模型地形或运行配置。
NX, NY = 720, 192
GRID_SPACING = 0.25
FIRST_LONGITUDE = 112.0
FIRST_LATITUDE = -23.5
OUTPUT_DIR = None  # None: 自动使用 data/preprocessed/gl12_quarter_<Nr>；或指定Path
OVERWRITE = True  # 允许重复运行：覆盖同名初始场、边界场和manifest；不删除其他文件
DRY_RUN = True  # True: 只检查输入并打印Nr、delR等信息，不生成文件；False: 生成数据
SMOOTH_SIGMA = 1.0  # 原生水平网格点
SMOOTH_THRESHOLD = 0.1
SOURCE_HALO_DEGREES = 1.0
INITIAL_MONTHS = [12, 1]

# 零起始边界；相邻 a,b 合并原生层 [a:b]，层数自动等于 len(LAYER_EDGES)-1。
# 从0开始、严格递增，末边界不超过原生层数；末边界之后的深层不使用。
LAYER_EDGES = [0, 5, 9, 12, 14] + list(range(16, 50))  # 当前38层方案
# LAYER_EDGES = [0, 4, 7, 10, 12] + list(range(14, 50))  # 40层
# LAYER_EDGES = list(range(51))                         # 原生50层，不合并
# 修改以上配置后直接运行脚本，无需命令行参数。
# ==============================================================

VARS = ('uo', 'vo', 'thetao', 'so', 'zos')
OBC_LETTERS = dict(uo='u', vo='v', thetao='t', so='s')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def vertical_grid(e3t, edges=None):
    """Each output layer merges native [edge[i], edge[i+1]) using e3t."""
    e3t = np.asarray(e3t, dtype=np.float64).squeeze()
    if e3t.ndim != 1 or not len(e3t) or not np.all(np.isfinite(e3t) & (e3t > 0)):
        raise ValueError('Expected a one-dimensional array of positive finite e3t thicknesses')
    if edges is None:
        edges = LAYER_EDGES
    raw = np.asarray(edges)
    if raw.ndim != 1 or len(raw) < 2 or raw.dtype.kind not in 'iu':
        raise ValueError('Layer edges must contain at least two integers')
    if raw[0] != 0 or np.any(np.diff(raw.astype(np.float64)) <= 0) or raw[-1] > len(e3t):
        raise ValueError('Layer edges must start at 0, strictly increase, and end <= native layer count')
    rid = raw.astype(np.int64)
    dz = np.array([e3t[a:b].sum() for a, b in zip(rid[:-1], rid[1:])])
    return rid, dz


def vertical_merge(a, e3t, rid):
    return np.stack([np.average(a[lo:hi], axis=0, weights=e3t[lo:hi])
                     for lo, hi in zip(rid[:-1], rid[1:])])


def largest_component(wet):
    labels, n = label(wet)  # four-neighbor connectivity; no artificial cuts
    if not n:
        raise ValueError('No wet surface cells')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax(), n


def nearest_fill(a):
    a = np.asarray(a, dtype=np.float64)
    good = np.isfinite(a)
    if not good.any():
        raise ValueError('No valid horizontal source values: cannot nearest-fill')
    if good.all():
        return a.copy(), 0
    indices = distance_transform_edt(~good, return_distances=False, return_indices=True)
    return a[tuple(indices)], int((~good).sum())


def smooth_sample_fill(a, basin, iy, ix, sigma=SMOOTH_SIGMA, threshold=SMOOTH_THRESHOLD,
                       cache=None):
    """Reuse mask-dependent weights and nearest indices when geometry is identical."""
    good = np.isfinite(a) & basin
    signature = (sigma, threshold)
    reuse = (cache is not None and cache.get('signature') == signature
             and np.array_equal(cache['good'], good)
             and np.array_equal(cache['iy'], iy) and np.array_equal(cache['ix'], ix))
    selection = np.ix_(iy, ix)
    if reuse:
        denominator = cache['denominator']
        valid = cache['valid']
        indices = cache['indices']
    else:
        denominator = gaussian_filter(good.astype(np.float64), sigma=sigma,
                                      mode='reflect')[selection]
        valid = (denominator > threshold) & basin[selection]
        if not valid.any():
            raise ValueError('No valid horizontal source values: cannot nearest-fill')
        indices = (None if valid.all() else
                   distance_transform_edt(~valid, return_distances=False, return_indices=True))
        if cache is not None:
            cache.update(signature=signature, good=good, iy=np.array(iy), ix=np.array(ix),
                         denominator=denominator, valid=valid, indices=indices)
    numerator = gaussian_filter(np.where(good, a, 0.), sigma=sigma, mode='reflect')[selection]
    sampled = np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan),
                        where=valid)
    return (sampled if indices is None else sampled[tuple(indices)]), int((~valid).sum())


def coordinate_indices(source, target, tolerance=1.e-3):
    source = np.asarray(source)
    if np.any(np.diff(source) <= 0):
        raise ValueError('Source coordinates must be strictly increasing')
    k = np.searchsorted(source, target)
    k = np.clip(k, 0, len(source)-1)
    prev = np.maximum(k-1, 0)
    k = np.where(abs(source[prev]-target) < abs(source[k]-target), prev, k)
    if np.max(abs(source[k]-target)) > tolerance:
        raise ValueError('Target points do not coincide with GL12 subsampling grid')
    if len(np.unique(k)) != len(target):
        raise ValueError('Repeated source coordinate index')
    return k


def write_binary(path, a, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(a)
    if not np.isfinite(a).all():
        raise ValueError(f'Nonfinite output: {path}')
    tmp = path.with_suffix(path.suffix+'.tmp')
    np.asarray(a, dtype='>f4').tofile(tmp); tmp.replace(path)
    if path.stat().st_size != a.size*4:
        raise ValueError('Binary size mismatch')
    # Independent on-disk round-trip, including signed values and all dimensions.
    reread = np.fromfile(path, dtype='>f4').reshape(a.shape)
    if not np.array_equal(reread, np.asarray(a, dtype=np.float32)):
        raise ValueError(f'Binary round-trip mismatch: {path}')
    records.append(dict(path=str(path), shape=list(a.shape), bytes=path.stat().st_size,
                        dtype='big-endian float32', sha256=sha256(path),
                        minimum=float(reread.min()), maximum=float(reread.max())))


def main():
    source_paths = dict(climatology=CLIMATOLOGY, coordinates=COORDINATES,
                        source_bathymetry=SOURCE_BATHYMETRY)
    if NX < 2 or NY < 2 or GRID_SPACING != .25:
        raise ValueError('Expected a quarter-degree grid with at least two points per direction')
    if not INITIAL_MONTHS or any(m not in range(1,13) for m in INITIAL_MONTHS):
        raise ValueError('INITIAL_MONTHS must contain calendar months 1..12')
    x = FIRST_LONGITUDE + np.arange(NX)*GRID_SPACING
    y = FIRST_LATITUDE + np.arange(NY)*GRID_SPACING
    with xr.open_dataset(COORDINATES) as d:
        e3t = np.asarray(d.e3t).squeeze().astype(np.float64)
    rid, dz = vertical_grid(e3t)
    nr = len(dz)
    print(f'Nr = {nr}; total depth = {dz.sum():.2f} m', flush=True)
    for start in range(0, nr, 10):
        prefix = ' delR= ' if start == 0 else '       '
        print(prefix + ','.join(f'{value:6.2f}' for value in dz[start:start+10]) + ',', flush=True)
    out = Path(OUTPUT_DIR or ROOT/f'data/preprocessed/gl12_quarter_{nr}').resolve()
    forbidden = [ROOT/'input', ROOT/'code', ROOT/'output', ROOT/'config']
    if any(out == x or x in out.parents for x in forbidden):
        raise ValueError('Use an independent output directory, not input/code/output/config')
    if not DRY_RUN and not OVERWRITE and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}; set OVERWRITE=True or choose another OUTPUT_DIR')
    with xr.open_dataset(CLIMATOLOGY) as source:
        if not set(VARS).issubset(source.data_vars) or source.sizes['depth'] != len(e3t):
            raise ValueError('Climatology must contain all five fields and match the e3t native layer count')
        months = source.time.dt.month.values
        if sorted(months.tolist()) != list(range(1,13)):
            raise ValueError('Expected exactly one climatology field for every calendar month')
        month_index = {m: int(np.flatnonzero(months == m)[0]) for m in range(1,13)}
        full_x = source.longitude.values; full_y = source.latitude.values
        # Full climatology-domain surface connectivity; source land topology only.
        with xr.open_dataset(SOURCE_BATHYMETRY) as d:
            query_x = xr.DataArray((full_x.astype(np.float64) + 180.) % 360. - 180., dims='x')
            query_y = xr.DataArray(full_y, dims='y')
            depth = d.deptho.sel(longitude=query_x, latitude=query_y,
                                method='nearest', tolerance=1.e-3).values
        surface_wet = np.isfinite(depth) & (depth > 0)
        basin_full, ncomponents = largest_component(surface_wet)
        # Keep >4*sigma source halo around target; connectivity was computed first.
        ix_crop = np.flatnonzero((full_x >= x[0]-SOURCE_HALO_DEGREES) & (full_x <= x[-1]+SOURCE_HALO_DEGREES))
        iy_crop = np.flatnonzero((full_y >= y[0]-SOURCE_HALO_DEGREES) & (full_y <= y[-1]+SOURCE_HALO_DEGREES))
        lon = full_x[ix_crop]; lat = full_y[iy_crop]
        ix = coordinate_indices(lon, x); iy = coordinate_indices(lat, y)
        basin = basin_full[np.ix_(iy_crop,ix_crop)]
        cropped = source.isel(latitude=slice(iy_crop[0],iy_crop[-1]+1),
                              longitude=slice(ix_crop[0],ix_crop[-1]+1))
        manifest = dict(status='metadata_validated', resolution_degrees=GRID_SPACING, nx=NX, ny=NY,
                        nr=nr, layer_group_edges=rid.tolist(), delR_m=dz.tolist(),
                        total_depth_m=float(dz.sum()), omitted_native_zero_based_layers=list(range(int(rid[-1]),len(e3t))),
                        months_order=list(range(1,13)), initial_months=INITIAL_MONTHS,
                        initial_mean_skipna=True, surface_connectivity='four-neighbor largest component; no A-star cut',
                        source_components=ncomponents, source_surface_removed=int((surface_wet & ~basin_full).sum()),
                        smoothing=dict(sigma_source_gridpoints=SMOOTH_SIGMA, normalized=True, threshold=SMOOTH_THRESHOLD, mode='reflect'),
                        missing_values='horizontal2D nearest per native layer/month after smoothing/subsampling; no vertical/time filling',
                        boundary_processing='extract S/N/W from the same complete processed monthly target fields',
                        velocity_coordinates='legacy common coordinate sampling; no additional C-grid half-cell relocation',
                        source_paths={k:str(path.resolve()) for k,path in source_paths.items()},
                        source_file_sizes={k:source_paths[k].stat().st_size for k in ['climatology','coordinates','source_bathymetry']},
                        source_units={v:source[v].attrs.get('units') for v in VARS},
                        longitude_centers=x.tolist(), latitude_centers=y.tolist(), fields=[])
        print(json.dumps({k:manifest[k] for k in ['nr','total_depth_m','source_components','source_surface_removed']},indent=2),flush=True)
        if DRY_RUN:
            print(f'DRY RUN: checks passed; output would be {out}; no files generated.', flush=True)
            return
        out.mkdir(parents=True, exist_ok=True)
        for sub in ['init','obcs']:(out/sub).mkdir(exist_ok=True)
        records = manifest['fields']
        try:
            for var in VARS:
                print('Processing',var,flush=True)
                is3d = var != 'zos'
                geometry_cache = [{} for _ in range(rid[-1] if is3d else 1)]
                def transform(raw, stage):
                    if not is3d:
                        field,nfill=smooth_sample_fill(raw,basin,iy,ix,cache=geometry_cache[0])
                        return field
                    sampled=np.empty((rid[-1],NY,NX),dtype=np.float64)
                    for k in range(rid[-1]):
                        try:sampled[k],nfill=smooth_sample_fill(raw[k],basin,iy,ix,cache=geometry_cache[k])
                        except ValueError as exc:raise ValueError(f'{var}, {stage}, native layer {k}: {exc}') from exc
                    return vertical_merge(sampled,e3t,rid)
                if is3d:
                    boundaries = dict(S=np.empty((12,nr,NX), dtype=np.float32),
                                      N=np.empty((12,nr,NX), dtype=np.float32),
                                      W=np.empty((12,nr,NY), dtype=np.float32))
                    for m in range(1,13):
                        a = cropped[var].isel(time=month_index[m], depth=slice(0,rid[-1]))
                        a = a.transpose('depth','latitude','longitude')
                        field = transform(np.asarray(a.values,dtype=np.float64),f'month_{m:02d}')
                        boundaries['S'][m-1] = field[:,0,:]
                        boundaries['N'][m-1] = field[:,-1,:]
                        boundaries['W'][m-1] = field[:,:,0]
                        print(var,'month',m,'/12',flush=True)
                    for boundary,arr in boundaries.items():
                        write_binary(out/f'obcs/OB{boundary}{OBC_LETTERS[var]}_4_{nr}.bin',arr,records)
                    del boundaries, field
                # All five initials average the configured native months BEFORE processing.
                a=cropped[var].isel(time=[month_index[m] for m in INITIAL_MONTHS])
                if is3d:a=a.isel(depth=slice(0,rid[-1]))
                a=a.mean('time',skipna=True)
                a=a.transpose(*(['depth','latitude','longitude'] if is3d else ['latitude','longitude']))
                initial=transform(np.asarray(a.values,dtype=np.float64),'initial_'+'_'.join(str(m) for m in INITIAL_MONTHS))
                write_binary(out/f'init/{var}_4_{nr}.bin',initial,records)
                del a,initial
            manifest['status']='complete'
        except Exception as exc:
            manifest['status']='failed';manifest['error']=repr(exc)
            raise
        finally:
            for r in records:r['path']=str(Path(r['path']).relative_to(out))
            (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE:',out,flush=True)


if __name__=='__main__':
    main()
