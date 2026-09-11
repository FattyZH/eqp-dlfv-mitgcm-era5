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
    """Open MDS output using run namelists for time and GGL90 metadata.

    Positive diagnostic frequencies use interval-midpoint timestamps. Multiple
    prefixes must have the same offset; otherwise open them separately. Prefixes
    absent from data.diagnostics (e.g. native U/T snapshots) are not shifted.
    """
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
        names = diag_list.get('filename', [])
        frequencies = diag_list.get('frequency', [])
        names = [names] if isinstance(names, str) else names
        frequencies = [frequencies] if np.isscalar(frequencies) else frequencies
        offsets = {
            name.strip(): int(freq / 2) if freq > 0 else 0
            for name, freq in zip(names, frequencies)
            if name is not None and freq is not None
        }
        prefixes = [prefix] if isinstance(prefix, str) else list(prefix)
        selected = {offsets.get(name, 0) for name in prefixes}
        if len(selected) > 1:
            raise ValueError("Prefixes have different diagnostic time offsets; open them separately")
        offset_sec = next(iter(selected), 0)
    if 'grid_vars_to_coords' not in kwargs:
        kwargs['grid_vars_to_coords']=False
    extra_variables = {**ex_vars, **kwargs.pop('extra_variables', {})}
    ds = xmitgcm.open_mdsdataset(path, prefix=prefix, extra_variables=extra_variables, **kwargs)
    fixed = {
        k: v.astype(v.dtype.newbyteorder('<'))
        if v.dtype.kind in 'fi' and v.dtype.byteorder == '>'
        else v
        for k, v in ds.coords.items()
    }
    fixed['time'] = ds["time"] - np.timedelta64(offset_sec, 's')
    ds = ds.assign_coords(fixed)
    return ds
