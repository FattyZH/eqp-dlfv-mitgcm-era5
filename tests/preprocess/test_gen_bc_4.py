"""Scientific invariants for the GL12 configurable-layer preprocessing."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import json
import xarray as xr
import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/preprocess/boundaries/gen_bc_4.py'
spec = importlib.util.spec_from_file_location('gen_bc_4', SCRIPT)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class PreprocessingTests(unittest.TestCase):
    def test_merge_conserves_retained_column_integral(self):
        for native_n, scheme in [(50, None), (6, [0, 2, 6]), (50, list(range(51)))]:
            with self.subTest(native_layers=native_n, edges=scheme):
                dz = np.arange(1., native_n+1.)
                edges, merged_dz = g.vertical_grid(dz, scheme)
                values = np.arange(native_n)[:, None, None] * np.ones((1, 2, 3))
                result = g.vertical_merge(values, dz, edges)
                np.testing.assert_allclose((result * merged_dz[:, None, None]).sum(0),
                                           (values[:edges[-1]] * dz[:edges[-1], None, None]).sum(0))
                self.assertEqual(len(result), len(edges)-1)
                self.assertEqual(merged_dz.sum(), dz[:edges[-1]].sum())

    def test_identity_groups_preserve_native_fields(self):
        values = np.arange(24.).reshape(4, 2, 3)
        edges, _ = g.vertical_grid([1., 2., 3., 4.], [0, 1, 2, 3, 4])
        np.testing.assert_equal(g.vertical_merge(values, np.arange(1., 5.), edges), values)

    def test_invalid_groups_rejected(self):
        for edges in [[1, 3], [0, 2, 2, 4], [0, 3, 2], [0, 5], [0],
                      [0, 1.5, 4], [False, True], [[0, 4]]]:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                g.vertical_grid(np.ones(4), edges)

    def test_connectivity_is_four_neighbor(self):
        wet = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=bool)
        retained, count = g.largest_component(wet)
        self.assertEqual(count, 3)
        self.assertEqual(retained.sum(), 2)
        self.assertFalse(retained[2, 0])

    def test_removed_basin_does_not_supply_values(self):
        basin = np.zeros((9, 9), bool)
        basin[:, :4] = True
        raw = np.where(basin, 7., 1.e6)
        output, _ = g.smooth_sample_fill(raw, basin, np.arange(9), np.arange(9))
        np.testing.assert_allclose(output, 7.)
        with self.assertRaises(ValueError):
            g.smooth_sample_fill(np.full((9, 9), np.nan), basin, np.arange(9), np.arange(9))

    def test_cached_geometry_matches_original_and_invalidates_on_missing_change(self):
        rng = np.random.default_rng(7)
        basin = rng.random((20,24)) > .2
        cache = {}
        iy, ix = np.arange(0,20,3), np.arange(0,24,3)
        for changed in [False, False, True]:
            raw = rng.normal(size=basin.shape)
            raw[~basin] = np.nan
            if changed:
                raw[:8,:8] = np.nan
            good = np.isfinite(raw) & basin
            numerator = g.gaussian_filter(np.where(good, raw, 0.), 1., mode='reflect')
            denominator = g.gaussian_filter(good.astype(float), 1., mode='reflect')
            old = np.divide(numerator, denominator, out=np.full(raw.shape,np.nan),where=denominator>.1)
            sampled = old[np.ix_(iy,ix)]
            sampled[~basin[np.ix_(iy,ix)]] = np.nan
            expected, count = g.nearest_fill(sampled)
            result, actual_count = g.smooth_sample_fill(raw,basin,iy,ix,cache=cache)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(actual_count, count)

    def test_nearest_fill_preserves_observed_cells(self):
        a = np.array([[1., np.nan, 3.], [np.nan, np.nan, 6.]])
        b, count = g.nearest_fill(a)
        self.assertEqual(count, 3)
        np.testing.assert_equal(b[np.isfinite(a)], a[np.isfinite(a)])
        self.assertTrue(np.isfinite(b).all())

    def test_generate_only_initials_and_boundaries_without_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            xy = np.arange(9)/12
            time = np.arange('2001-01', '2002-01', dtype='datetime64[M]')
            values = np.broadcast_to(np.arange(12)[:,None,None,None] +
                                     np.arange(6)[None,:,None,None]*10., (12,6,9,9)).copy()
            fields = {v: (('time','depth','latitude','longitude'), values)
                      for v in ['uo','vo','thetao','so']}
            fields['zos'] = (('time','latitude','longitude'), values[:,0])
            xr.Dataset(fields, coords=dict(time=time,depth=np.arange(6),latitude=xy,longitude=xy)).to_netcdf(d/'clim.nc')
            xr.Dataset({'e3t': ('depth', np.ones(6))}).to_netcdf(d/'coordinates.nc')
            xr.Dataset({'deptho': (('latitude','longitude'), np.ones((9,9))*100)},
                       coords=dict(latitude=xy,longitude=xy)).to_netcdf(d/'bathy.nc')
            settings = dict(CLIMATOLOGY=d/'clim.nc', COORDINATES=d/'coordinates.nc',
                            SOURCE_BATHYMETRY=d/'bathy.nc', OUTPUT_DIR=d/'result',
                            NX=3,NY=3,FIRST_LONGITUDE=0.,FIRST_LATITUDE=0.,
                            LAYER_EDGES=[0,2,6],DRY_RUN=False,OVERWRITE=True)
            with patch.multiple(g, **settings), contextlib.redirect_stdout(io.StringIO()):
                g.main()
                marker = d/'result'/'unrelated.txt'
                marker.write_text('keep')
                g.main()  # Re-run into existing init/obcs directories.
                self.assertEqual(marker.read_text(), 'keep')
                marker.unlink()
                with patch.object(g, 'OVERWRITE', False), self.assertRaises(FileExistsError):
                    g.main()
            out = d/'result'
            self.assertEqual({p.name for p in out.iterdir()}, {'init','obcs','manifest.json'})
            self.assertEqual(len(list(out.rglob('*.bin'))), 17)
            initial = np.fromfile(out/'init/uo_4_2.bin', '>f4').reshape(2,3,3)
            np.testing.assert_allclose(initial[:,0,0], [10.5,40.5])
            for side, length in [('S',3),('N',3),('W',3)]:
                boundary = np.fromfile(out/f'obcs/OB{side}u_4_2.bin','>f4').reshape(12,2,length)
                np.testing.assert_allclose(boundary[:,:,0], np.arange(12)[:,None]+[5.,35.])
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'], 'complete')

    def test_binary_month_depth_horizontal_order(self):
        a = np.arange(2*3*4).reshape(2, 3, 4) - 10.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'boundary.bin'
            records = []
            g.write_binary(p, a, records)
            np.testing.assert_array_equal(np.fromfile(p, '>f4'), a.ravel())
            self.assertEqual(p.stat().st_size, a.size*4)


if __name__ == '__main__':
    unittest.main()
