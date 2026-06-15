import xmitgcm
from os.path import join,exists
import numpy as np
import f90nml

ex_vars = {
    'GGL90viscArU': dict(dims=['k_l', 'j', 'i_g'], attrs=dict(
        standard_name="GGL90_vertical_eddy_visc_U",
        long_name='GGL90 vertical eddy viscosity coefficient for U',
        units='m2 s-1')),
    'GGL90viscArV': dict(dims=['k_l', 'j_g', 'i'], attrs=dict(
        standard_name="GGL90_vertical_eddy_visc_V",
        long_name='GGL90 vertical eddy viscosity coefficient for V',
        units='m2 s-1')),
    'GGL90diffKr': dict(dims=['k_l', 'j', 'i'], attrs=dict(
        standard_name="GGL90_diff_tracer",
        long_name='GGL90 vertical diffusion coefficient for tracers',
        units='m2 s-1')),
}

def open_mds(path, prefix=None, **kwargs):
    config_path = {
        'data':join(path,'data'),
        'cal':join(path,'data.cal'),
        'diag':join(path,'data.diagnostics'),
    }
    if exists(config_path['cal']) and 'ref_date' not in kwargs:
        data = f90nml.read(config_path['cal'])
        c = data['CAL_NML']
        d1, d2 = c['startDate_1'], c.get('startDate_2', 0)
        s = f"{d1:08d}{d2:06d}"
        kwargs['ref_date'] = f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}"
    if exists(config_path['data']) and 'delta_t' not in kwargs:
        data = f90nml.read(config_path['data'])
        kwargs['delta_t'] = data['PARM03']['deltaT']
    offset_sec = 0
    if exists(config_path['diag']) and prefix:
        diag_list = f90nml.read(config_path['diag']).get("DIAGNOSTICS_LIST", {})
        diag_list['filename']
        ind = diag_list['filename'].index(prefix)
        freq = diag_list['frequency'][ind]
        offset_sec = int(freq / 2) if freq > 0 else 0
    if 'grid_vars_to_coords' not in kwargs:
        kwargs['grid_vars_to_coords']=False
    ds = xmitgcm.open_mdsdataset(path, prefix=prefix,extra_variables=ex_vars, **kwargs)
    fixed = {
        k: v.astype(v.dtype.newbyteorder('<'))
        if v.dtype.kind in 'fi' and v.dtype.byteorder == '>'
        else v
        for k, v in ds.coords.items()
    }
    fixed['time'] = ds["time"] - np.timedelta64(offset_sec, 's')
    ds = ds.assign_coords(fixed)
    return ds
