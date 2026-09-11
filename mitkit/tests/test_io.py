"""Regression checks for the retained I/O helpers; no model output is written."""
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from mitkit.io import open_mds, parse_diag


class TestOpenMds(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').write_text('&PARM03\n deltaT=450.,\n&\n')
        (self.root / 'data.cal').write_text('&CAL_NML\n startDate_1=20000101,\n&\n')
        self.ds = xr.Dataset(coords={'time': np.array(['2000-01-01T02:00:00'], dtype='datetime64[s]')})

    def run_open(self, prefix, **kwargs):
        module = importlib.import_module('mitkit.io.open_mds')
        with patch.object(module.xmitgcm, 'open_mdsdataset', return_value=self.ds) as loader:
            result = open_mds(self.root, prefix=prefix, **kwargs)
        return result, loader.call_args.kwargs

    def test_single_diagnostic_and_metadata(self):
        (self.root / 'data.diagnostics').write_text("&DIAGNOSTICS_LIST\n filename='diag3d', frequency=3600.,\n&\n")
        result, kwargs = self.run_open('diag3d', extra_variables={'custom': {}})
        self.assertEqual(result.time.values[0], np.datetime64('2000-01-01T01:30:00'))
        self.assertEqual(kwargs['delta_t'], 450.)
        self.assertEqual(kwargs['ref_date'], '2000-01-01 00:00:00')
        self.assertIn('custom', kwargs['extra_variables'])
        self.assertIn('GGL90viscArU', kwargs['extra_variables'])

    def test_snapshot_prefix_list(self):
        (self.root / 'data.diagnostics').write_text("&DIAGNOSTICS_LIST\n filename='diag3d', frequency=3600.,\n&\n")
        result, kwargs = self.run_open(['U', 'T'])
        np.testing.assert_array_equal(result.time, self.ds.time)
        self.assertEqual(kwargs['prefix'], ['U', 'T'])

    def test_matching_offsets(self):
        (self.root / 'data.diagnostics').write_text("&DIAGNOSTICS_LIST\n filename='a','b', frequency=3600.,3600.,\n&\n")
        result, _ = self.run_open(['a', 'b'])
        self.assertEqual(result.time.values[0], np.datetime64('2000-01-01T01:30:00'))

    def test_mixed_offsets_rejected(self):
        (self.root / 'data.diagnostics').write_text("&DIAGNOSTICS_LIST\n filename='a','b', frequency=3600.,-3600.,\n&\n")
        with self.assertRaisesRegex(ValueError, 'different diagnostic time offsets'):
            self.run_open(['a', 'b'])

    def test_explicit_time_settings(self):
        _, kwargs = self.run_open('U', ref_date='1990-01-01', delta_t=10)
        self.assertEqual(kwargs['ref_date'], '1990-01-01')
        self.assertEqual(kwargs['delta_t'], 10)


class TestParseDiag(unittest.TestCase):
    HEADER = '# frequency: 3600\n# phase: 0\n# regions: 0\n# fields: momKE\n# nb of lev: 1\n# end of header\n'

    def parse(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stat.txt'
            path.write_text(text)
            return parse_diag(path, ref_date='2000-01-01')

    def test_complete_record(self):
        result = self.parse(self.HEADER + 'field : momKE ; Iter = 0 ; region # 0 ; nb.Lev = 1\nk ave std min max vol\n0 1 0.1 0 2 10\n# records End here\n')
        self.assertTrue(result['complete'])
        self.assertEqual(result['momKE'].shape, (1, 1, 5))
        np.testing.assert_array_equal(result['iter'], [0])

    def test_missing_header_terminator(self):
        with self.assertRaisesRegex(ValueError, 'missing.*end of header'):
            self.parse('# frequency: 3600\n')

    def test_missing_header_fields(self):
        with self.assertRaisesRegex(ValueError, 'missing header fields'):
            self.parse('# end of header\n')

    def test_bad_record(self):
        with self.assertRaisesRegex(ValueError, 'malformed diagnostic record'):
            self.parse(self.HEADER + 'invalid record\n')
