#!/usr/bin/env python

import json
import logging
import mmap
import operator
import requests
import shutil
import unittest
from collections import namedtuple
from functools import reduce
from tempfile import mkdtemp, NamedTemporaryFile

import numpy as np
import six
from six import BytesIO, iteritems, itervalues
from numpy.testing import assert_allclose, assert_equal

try:
    from unittest import mock
except ImportError:
    import mock

from curves import OSCCurve, SNFiles, NoPhotometryError, EmptyPhotometryError


TRIANGLE_JSON_PATH = 'test_triangle.json'
BINNING_JSON_PATH = 'test_binning.json'

SNS_NO_CMAIMED_TYPE = frozenset((
    'SNLS-03D3ce',
))

SNS_UPPER_LIMIT = frozenset((
    'SNLS-04D3fq',
    'PS1-10ahf',
    'MLS121209:093512+152855',
))

SNS_E_LOWER_UPPER_MAGNITUDE = frozenset((
    'SNLS-04D3fq',
))

SNE_E_TIME = frozenset((
    'MLS121209:093512+152855',
))

SNS_UNORDERED_PHOTOMETRY = frozenset((
    'PTF09atu',
    'PS1-10ahf',
))

SNS_HAVE_ZERO_E_MAGNITUDE = frozenset((
    'Gaia14ado',
))

SNS_HAVE_NOT_PHOTOMETRY = frozenset((
    'GRB 081025A',
))

SNS_HAVE_NOT_MAGN_ERRORS = frozenset((
    'SN2005V',
))

SNS_ZERO_VALID_PHOTOMETRY_DOTS = frozenset((
    'SN2007bk',
))

SNS_HAVE_SPECTRA = frozenset((
    'SNLS-04D3fq',
))

SNS_HAVE_NOT_SPECTRA = frozenset((
    'SN1993A',
))

SNS_HAVE_B_BAND = frozenset((
    'SN1993A',
))

SNS_ALL = frozenset.union(SNS_NO_CMAIMED_TYPE, SNS_UPPER_LIMIT, SNS_E_LOWER_UPPER_MAGNITUDE, SNE_E_TIME,
                          SNS_UNORDERED_PHOTOMETRY, SNS_HAVE_ZERO_E_MAGNITUDE, SNS_HAVE_B_BAND)
SNS_ALL_TUPLE = tuple(sorted(SNS_ALL))


def _get_curves(sns, *args, **kwargs):
    return [OSCCurve.from_name(sn, *args, **kwargs) for sn in sns]


class BasicSNFilesTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = mkdtemp(prefix='tmpsne')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _check_file_is_json(self, fpath):
        with open(fpath) as fp:
            try:
                json.load(fp)
            except json.JSONDecodeError as e:
                self.fail(str(e))

    def _check_file_contains_SN_name(self, fpath, snname):
        with open(fpath, 'r+') as fp:
            s = mmap.mmap(fp.fileno(), 0)
            if six.PY2:
                snname_bytes = snname
            else:
                snname_bytes = snname.encode('utf-8')
            try:
                self.assertNotEqual(s.find(snname_bytes), -1,
                                    msg="File {} doesn't contain name of SN '{}'".format(fpath, snname))
            finally:
                s.close()

    def check_SNfiles(self, sn_files):
        for i, fpath in enumerate(sn_files.filepaths):
            self._check_file_is_json(fpath)
            self._check_file_contains_SN_name(fpath, sn_files[i])


class DownloadFromSneTestCase(BasicSNFilesTestCase):
    def test_list_download(self):
        sn_files = SNFiles(SNS_ALL, path=self.tmp_dir, offline=False)
        self.check_SNfiles(sn_files)

    def test_not_found_raises(self):
        snname = '8cbac453'
        with self.assertRaises(requests.HTTPError):
            SNFiles([snname], path=self.tmp_dir, offline=False)


class DownloadCacheTestCase(BasicSNFilesTestCase):
    etag1 = b'1'
    etag2 = b'2'
    content = b'1'

    def response_template(self):
        r = requests.Response()
        r.status_code = 200
        r.raw = BytesIO(self.content)
        return r

    def setUp(self):
        super(DownloadCacheTestCase, self).setUp()
        self.snnames = SNS_ALL_TUPLE[:1]

    @mock.patch.object(SNFiles, '_get_response')
    @mock.patch.object(SNFiles, '_set_file_etag')
    @mock.patch.object(SNFiles, '_get_file_etag', return_value=etag1)
    def test_download_after_etag_update(self, mock_get_file_etag, mock_set_file_etag, mock_get_response):
        response = self.response_template()
        response.headers = {'etag': self.etag2}
        mock_get_response.return_value = response

        def set_file_etag_side_effect(fpath, etag):
            self.assertEqual(etag, self.etag2)
        mock_set_file_etag.side_effect = set_file_etag_side_effect

        SNFiles(self.snnames, path=self.tmp_dir, offline=False)
        self.assertTrue(mock_get_file_etag.called)
        self.assertTrue(mock_get_response.called)
        self.assertTrue(mock_set_file_etag.called)

    @mock.patch.object(SNFiles, '_get_response')
    @mock.patch.object(SNFiles, '_set_file_etag')
    @mock.patch.object(SNFiles, '_get_file_etag', return_value=etag1)
    def test_not_download_for_same_etag(self, mock_get_file_etag, mock_set_file_etag, mock_get_response):
        response = self.response_template()
        response.status_code = 304
        response.iter_content = mock.Mock()
        mock_get_response.return_value = response

        def get_response_side_effect(url, etag):
            self.assertEqual(etag, self.etag1)
            return mock.DEFAULT
        mock_get_response.side_effect = get_response_side_effect

        SNFiles(self.snnames, path=self.tmp_dir, offline=False)
        self.assertTrue(mock_get_file_etag.called)
        self.assertTrue(mock_get_response.called)
        response.iter_content.assert_not_called()
        mock_set_file_etag.assert_not_called()


    @mock.patch.object(SNFiles, '_get_response')
    @mock.patch.object(SNFiles, '_get_file_etag', return_value=None)
    def test_download_if_no_xatrr(self, mock_get_file_etag, mock_get_response):
        mock_get_response.return_value = self.response_template()

        SNFiles(self.snnames, path=self.tmp_dir, offline=False)
        self.assertTrue(mock_get_file_etag.called)
        self.assertTrue(mock_get_response.called)


class SNFilesOfflineMode(BasicSNFilesTestCase):
    def setUp(self):
        super(SNFilesOfflineMode, self).setUp()
        self.snnames_exist = SNS_ALL_TUPLE[:1]
        self.snnames_not_exist = ['c4048589']
        self.sn_files_online = SNFiles(self.snnames_exist, path=self.tmp_dir, offline=False)

    def test_file_exists(self):
        sn_files = SNFiles(self.snnames_exist, path=self.tmp_dir, offline=True)
        self.check_SNfiles(sn_files)

    def test_raises_if_no_file(self):
        with self.assertRaises(ValueError):
            SNFiles(self.snnames_not_exist, path=self.tmp_dir, offline=True)


class LoadSNListFromCSV(unittest.TestCase):
    def test_csv_download(self):
        with NamedTemporaryFile(mode='w+', suffix='.csv') as fd:
            with open(fd.name, 'w') as f:
                f.write('Name\n')
                f.write('\n'.join(SNS_ALL))
            sn_files = SNFiles(fd.name)
            self.assertEqual(set(sn_files), SNS_ALL)


class SNFilesReprTestCase(unittest.TestCase):
    def test_repr(self):
        sn_files = SNFiles(SNS_ALL)
        assertRegexp = self.assertRegexpMatches if six.PY2 else self.assertRegex
        assertRegexp(repr(sn_files), r'SN names:.+')


class ReadLightCurvesFromJsonTestCase(unittest.TestCase):
    def setUp(self):
        self.sn_files = SNFiles(SNS_ALL)

    def test_download_and_read(self):
        for fname in self.sn_files.filepaths:
            OSCCurve.from_json(fname)


class AddPhotometryDotTestCase(unittest.TestCase):
    def setUp(self):
        self.band = 'B'
        self.curves = _get_curves(SNS_HAVE_B_BAND, bands=self.band)

    def test_add_dot_to_msd(self):
        x = np.array([-100])
        y = np.array([123])
        err = np.array([1.23])
        dot = np.rec.array([x, y, err], names=('x', 'y', 'err'))
        dots = {'B': dot}
        for curve in self.curves:
            msd = curve.multi_state_data()
            msd.append_dict(dots)
            self.assertIn(x, msd.odict[self.band].x)
            self.assertIn(y, msd.odict[self.band].y)
            self.assertIn(err, msd.odict[self.band].err)
            self.assertIn(x, msd.arrays.x[:,1])

    def test_msd_with_zero_dots(self):
        for curve in self.curves:
            x = np.array([0, 1])
            msd = curve.msd_with_zero_valued_dots(x)
            self.assertIn(x[0], msd.odict[self.band].x)
            self.assertIn(x[1], msd.odict[self.band].x)
            self.assertIn(0, msd.odict[self.band].y)
            self.assertIn(0, msd.odict[self.band].err)
            self.assertIn(x[0], msd.arrays.x[:, 1])
            self.assertIn(x[1], msd.arrays.x[:, 1])

class BandDataTestCase(unittest.TestCase):
    def test_has_band(self):
        band = 'B'
        for sn in SNS_HAVE_B_BAND:
            curve = OSCCurve.from_name(sn, bands=band)
            self.assertEqual(len(curve.bands), 1)
            self.assertIn(band, curve)

    def test_has_not_band(self):
        band = 'HBRW'
        sn = SNS_ALL_TUPLE[0]
        with self.assertRaises(ValueError):
            OSCCurve.from_name(sn, bands=band)


class UpperLimitTestCase(unittest.TestCase):
    def setUp(self):
        self.curves = _get_curves(SNS_UPPER_LIMIT)

    def test_has_upper_limit(self):
        for curve in self.curves:
            has_upper_limit = reduce(operator.__or__,
                                     (np.any(lc['isupperlimit']) for lc in curve.odict.values()))
            self.assertTrue(has_upper_limit, 'SN {} light curves should have upper limit dots')


class EpsilonTimeTestCase(unittest.TestCase):
    def setUp(self):
        self.curves = _get_curves(SNE_E_TIME)

    def test_has_finite_e_time(self):
        for curve in self.curves:
            has_finite_e_time = reduce(operator.__or__,
                                       (np.any(np.isfinite(lc['err_x'])) for lc in curve.odict.values()))
            self.assertTrue(has_finite_e_time,
                            'SN {} should have finite e_time dots')


class BadPhotometryTestCase(unittest.TestCase):
    def test_has_not_photometry_field(self):
        sn_files = SNFiles(SNS_HAVE_NOT_PHOTOMETRY)
        for fpath in sn_files.filepaths:
            with self.assertRaises(NoPhotometryError,
                                   msg='{} should not contain photometry and KeyError should be raised'.format(fpath)):
                OSCCurve.from_json(fpath)

    def test_has_zero_valid_photometry_dots(self):
        sn_files = SNFiles(SNS_ZERO_VALID_PHOTOMETRY_DOTS)
        for fpath in sn_files.filepaths:
            with self.assertRaises(EmptyPhotometryError,
                                   msg='{} should not contain any valid photometry dot'.format(fpath)):
                OSCCurve.from_json(fpath)

    def test_has_not_observations_for_the_band(self):
        band = 'MDUzZDJ'
        for snname in SNS_ALL_TUPLE:
            with self.assertRaises(EmptyPhotometryError,
                                   msg='{} should not contain any observations in the band {}'.format(snname, band)):
                OSCCurve.from_name(snname, bands=band)

    def test_has_not_magn_errors(self):
        curves = _get_curves(SNS_HAVE_NOT_MAGN_ERRORS)
        for curve in curves:
            with self.assertRaises(EmptyPhotometryError,
                                   msg='{} should not contain finite errors'.format(curve.name)):
                curve.filtered(with_inf_e_flux=False)


class HasSpectraTestCase(unittest.TestCase):
    def test_has_spectra(self):
        curves = _get_curves(SNS_HAVE_SPECTRA)
        for curve in curves:
            self.assertTrue(curve.has_spectra, '{} data should contain spectra'.format(curve.name))

    def test_has_not_spectra(self):
        curves = _get_curves(SNS_HAVE_NOT_SPECTRA)
        for curve in curves:
            self.assertFalse(curve.has_spectra, '{} data should not contain spectra'.format(curve.name))


class TemporalOrderTestCase(unittest.TestCase):
    def setUp(self):
        self.sn_files = SNFiles(SNS_UNORDERED_PHOTOMETRY)

    def test_order(self):
        for fpath in self.sn_files.filepaths:
            curve = OSCCurve.from_json(fpath)
            for lc in curve.values():
                self.assertTrue(np.all(np.diff(lc['x']) >= 0))

    @unittest.skipIf(six.PY2, 'Logging testing is missed in Python 2')
    def test_unordered_logging(self):
        for fpath in self.sn_files.filepaths:
            with self.assertLogs(level=logging.INFO):
                OSCCurve.from_json(fpath)


class HasZeroEMagnitudeTestCase(unittest.TestCase):
    def setUp(self):
        self.curves = _get_curves(SNS_HAVE_ZERO_E_MAGNITUDE)

    def test_has_infinite_errors(self):
        for curve in self.curves:
            for blc in itervalues(curve):
                if np.any(~np.isfinite(blc['err'])):
                    break
            else:
                assert False, '{} data should contain infinite errors'

    def test_have_only_positive_errors(self):
        for curve in self.curves:
            for band, blc in iteritems(curve):
                e_flux = blc['err']
                self.assertTrue(np.all(e_flux[np.isfinite(e_flux)] > 0),
                                msg='{} in band {} has non-positive errors'.format(curve.name, band))


class LightCurveLearningDataTestCase(unittest.TestCase):
    def setUp(self):
        with open(TRIANGLE_JSON_PATH) as fp:
            self.json_data = json.load(fp)['test_triangle']
        self.time = np.arange(0, 70, 10)
        self.u_magn = np.r_[5:-1:-3,-1:6.5:1.5]

    def test_X_U(self):
        curve = OSCCurve(self.json_data, bands='U')
        assert_equal(curve.X, np.stack((np.zeros_like(self.time), self.time), axis=1))

    def test_X_states(self):
        curve = OSCCurve(self.json_data)
        assert_equal(curve.X[:len(self.time),0], np.zeros_like(self.time))
        assert_equal(curve.X[len(self.time):,0], np.ones_like(self.time))

    def test_y_U(self):
        curve = OSCCurve(self.json_data, bands='U')
        assert_allclose(-2.5*np.log10(curve.y*curve.norm), self.u_magn)


class LightCurveBinningTestCase(unittest.TestCase):
    def setUp(self):
        self.band = 'X'
        self.time = np.arange(9, dtype=np.int)
        self.bin_width = 1
        osc_curve = OSCCurve.from_json(BINNING_JSON_PATH)
        self.curve = osc_curve[self.band]
        self.binned_curve = osc_curve.binned(bin_width=1, discrete_time=False)[self.band]
        self.binned_curve_discrete = osc_curve.binned(bin_width=1, discrete_time=True)[self.band]

    def sample_and_binned(self, t):
        sample_idx = (self.curve['x'] >= t) & (self.curve['x'] < t+1)
        sample = self.curve[sample_idx]
        binned = self.binned_curve[t]
        return sample, binned

    def test_all_discrete_time(self):
        assert_allclose(self.time + 0.5 * self.bin_width, self.binned_curve_discrete['x'])
        assert_allclose(0.5 * self.bin_width, self.binned_curve_discrete['err_x'])

    def test_all_upper_limits(self):
        t = self.time[0]
        sample, binned = self.sample_and_binned(t)
        expected = sample[np.argmin(sample['y'])]

        assert_allclose(expected['x'], binned['x'])
        assert_allclose(expected['err_x'], binned['err_x'])
        self.assertEqual(expected['y'], binned['y'])
        assert_allclose(expected['err'], binned['err'], rtol=0, atol=0, equal_nan=True)
        self.assertTrue(binned['isupperlimit'])

    def test_all_without_errors(self):
        t = self.time[1]
        sample, binned = self.sample_and_binned(t)
        expected_time = np.mean(sample['x'])
        expected_e_time = np.std(sample['x']) / np.sqrt(sample.size - 1)
        expected_flux = np.mean(sample['y'])

        assert_allclose(expected_time, binned['x'])
        assert_allclose(expected_e_time, binned['err_x'])
        assert_allclose(expected_flux, binned['y'])
        self.assertTrue(np.isnan(binned['err']))
        self.assertFalse(binned['isupperlimit'])

    def test_all_close_with_errors(self):
        t = self.time[2]
        sample, binned = self.sample_and_binned(t)

        # Check test data
        for i, dot1 in enumerate(sample[:-1]):
            for dot2 in sample[i+1:]:
                d1, d2 = sorted([dot1, dot2], key=lambda d: d['err'])
                lower = d2['y'] - d2['err']
                upper = d2['y'] + d2['err']
                self.assertTrue(lower <= d1['y'] - d1['err'] <= upper
                                or lower <= d1['y'] + d1['err'] <= upper)
        assert_allclose(np.diff(sample['err']), 0, atol=1e-2*np.min(sample['err']))

        expected_time = np.mean(sample['x'])
        expected_e_time = np.std(sample['x']) / np.sqrt(sample.size - 1)
        expected_flux = np.mean(sample['y'])
        expected_min_err = 1 / np.sum(1 / sample['err']**2)

        assert_allclose(expected_time, binned['x'])
        assert_allclose(expected_e_time, binned['err_x'])
        assert_allclose(expected_flux, binned['y'])
        assert_allclose(1.0000000000000002, binned['y'])
        self.assertLessEqual(expected_min_err, binned['err'])
        assert_allclose(0.0866025403784, binned['err'])
        self.assertFalse(binned['isupperlimit'])
        for dot1 in sample:
            d1, d2 = sorted([dot1, binned], key=lambda d: d['err'])
            lower = d2['y'] - d2['err']
            upper = d2['y'] + d2['err']
            self.assertTrue(lower <= d1['y'] - d1['err'] <= upper
                            or lower <= d1['y'] + d1['err'] <= upper)

    def test_close_with_errors_and_one_bad(self):
        t = self.time[3]
        sample, binned = self.sample_and_binned(t)
        close_dots = sample[sample['y'] < 3]  # flux of the bad should be 10
        self.assertEqual(close_dots.size + 1, sample.size)
        bad = sample[sample['y'] > 3][0]
        # Check test data
        for i, dot1 in enumerate(sample[:-1]):
            for dot2 in sample[i+1:]:
                d1, d2 = sorted([dot1, dot2], key=lambda d: d['err'])
                lower = d2['y'] - d2['err']
                upper = d2['y'] + d2['err']
                self.assertTrue(lower <= d1['y'] - d1['err'] <= upper
                                or lower <= d1['y'] + d1['err'] <= upper)

        expected_time = np.average(sample['x'], weights=1/sample['err']**2)
        expected_min_e_time = np.std(sample['x']) / np.sqrt(sample.size - 1)
        expected_min_flux = np.mean(close_dots['y'])
        expected_max_flux = bad['y']
        expected_min_err1 = 1 / np.sum(1 / close_dots['err']**2)
        expected_min_err2 = np.sqrt(np.sum((sample['y'] - np.mean(sample['y']))**2) / (sample.size - 1))
        expected_min_err = min(expected_min_err1, expected_min_err2)

        assert_allclose(expected_time, binned['x'])
        self.assertLess(expected_min_e_time, binned['err_x'])
        self.assertLessEqual(expected_min_flux, binned['y'])
        assert_allclose(1.0013294434801876, binned['y'])
        self.assertGreaterEqual(expected_max_flux, binned['y'])
        assert_allclose(0.0971313820607, binned['err'])
        self.assertLessEqual(expected_min_err, binned['err'])
        self.assertFalse(binned['isupperlimit'])
        for dot1 in close_dots:
            d1, d2 = sorted([dot1, binned], key=lambda d: d['err'])
            lower = d2['y'] - d2['err']
            upper = d2['y'] + d2['err']
            self.assertTrue(lower <= d1['y'] - d1['err'] <= upper
                            or lower <= d1['y'] + d1['err'] <= upper)

    def test_two_groups_with_equal_errors(self):
        t = self.time[4]
        sample, binned = self.sample_and_binned(t)
        # Flux of lower group is ~ 1, fkux of upper group is ~ 10
        lower_group = sample[sample['y'] < 3]
        upper_group = sample[sample['y'] > 3]
        # Check test data
        for group in (lower_group, upper_group):
            for i, dot1 in enumerate(group[:-1]):
                for dot2 in group[i+1:]:
                    d1, d2 = sorted([dot1, dot2], key=lambda d: d['err'])
                    lower = d2['y'] - d2['err']
                    upper = d2['y'] + d2['err']
                    self.assertTrue(lower <= d1['y'] - d1['err'] <= upper
                                    or lower <= d1['y'] + d1['err'] <= upper)
        assert_allclose(np.diff(sample['err']), 0, atol=1e-2 * np.min(sample['err']))

        expected_time = np.mean(sample['x'])
        expected_e_time = np.std(sample['x']) / np.sqrt(sample.size - 1)
        expected_flux = np.mean(sample['y'])
        expected_min_flux = np.mean(lower_group['y'])
        expected_max_flux = np.mean(upper_group['y'])
        expected_err = np.sqrt(np.sum((sample['y']-np.mean(sample['y']))**2) / sample.size / (sample.size-1))

        assert_allclose(expected_time, binned['x'])
        assert_allclose(expected_e_time, binned['err_x'])
        assert_allclose(expected_flux, binned['y'])
        assert_allclose(5.500000000000001, binned['y'])
        self.assertLess(expected_min_flux, binned['y'])
        self.assertGreater(expected_max_flux, binned['y'])
        assert_allclose(expected_err, binned['err'])
        assert_allclose(2.01279242182, binned['err'])
        self.assertFalse(binned['isupperlimit'])

    def test_upper_limit_and_without_error(self):
        t = self.time[5]
        sample, binned = self.sample_and_binned(t)
        self.assertEqual(sample.size, 2)
        upper_limit = sample[sample['isupperlimit']]
        self.assertEqual(upper_limit.size, 1)
        upper_limit = upper_limit[0]
        dot = sample[~sample['isupperlimit']][0]
        # Check test data
        self.assertLess(upper_limit['y'], dot['y'])

        self.assertEqual(dot['x'], binned['x'])
        assert_allclose(dot['err_x'], binned['err_x'], equal_nan=True)
        self.assertEqual(dot['y'], binned['y'])
        self.assertTrue(np.isnan(binned['err']))
        self.assertFalse(binned['isupperlimit'])

    def test_upper_limit_and_with_error(self):
        t = self.time[6]
        sample, binned = self.sample_and_binned(t)
        self.assertEqual(sample.size, 2)
        upper_limit = sample[sample['isupperlimit']]
        self.assertEqual(upper_limit.size, 1)
        dot = sample[~sample['isupperlimit']][0]

        self.assertTrue(dot['x'], binned['x'])
        assert_allclose(dot['err_x'], binned['err_x'], equal_nan=True)
        self.assertEqual(dot['y'], binned['y'])
        self.assertEqual(dot['err'], binned['err'])
        self.assertFalse(binned['isupperlimit'])

    def test_without_error_and_with_error(self):
        t = self.time[7]

        sample, binned = self.sample_and_binned(t)
        self.assertEqual(sample.size, 2)
        without_error = sample[~np.isfinite(sample['err'])]
        self.assertEqual(without_error.size, 1)
        without_error = without_error[0]
        dot = sample[np.isfinite(sample['err'])]
        self.assertEqual(dot.size, 1)
        dot = dot[0]

        self.assertTrue(dot['x'], binned['x'])
        assert_allclose(dot['err_x'], binned['err_x'], equal_nan=True)
        self.assertEqual(dot['y'], binned['y'])
        self.assertEqual(dot['err'], binned['err'])
        self.assertFalse(binned['isupperlimit'])

    def test_upper_limit_without_error_with_error(self):
        t = self.time[8]
        sample, binned = self.sample_and_binned(t)
        self.assertEqual(sample.size, 3)
        self.assertEqual(np.sum(sample['isupperlimit']), 1)
        self.assertEqual(np.sum(~sample['isupperlimit'] & ~np.isfinite(sample['err'])), 1)
        dot = sample[~sample['isupperlimit'] & np.isfinite(sample['err'])][0]

        self.assertTrue(dot['x'], binned['x'])
        assert_allclose(dot['err_x'], binned['err_x'], equal_nan=True)
        self.assertEqual(dot['y'], binned['y'])
        self.assertEqual(dot['err'], binned['err'])
        self.assertFalse(binned['isupperlimit'])
