#!/usr/bin/env python

import json
import logging
import mmap
import operator
import requests
import shutil
import unittest
from functools import reduce
from tempfile import mkdtemp, NamedTemporaryFile

import numpy as np
import pandas
import six
from numpy.testing import assert_allclose, assert_equal

from curves import SNCurve, SNFiles


TEST_JSON_PATH = 'test.json'

SNS_NO_CMAIMED_TYPE = frozenset((
    'SNLS-03D3ce',
))

SNS_UPPER_LIMIT = frozenset((
    'SNLS-04D3fq',
    'PS1-10ahf',
))

SNS_E_LOWER_UPPER_MAGNITUDE = frozenset((
    'SNLS-04D3fq',
))

SNS_UNORDERED_PHOTOMETRY = frozenset((
    'PTF09atu',
    'PS1-10ahf',
))

SNS_HAS_SPECTRA = frozenset((
    'SNLS-04D3fq',
))

SNS_HAS_NOT_SPECTRA = frozenset((
    'SN1993A',
))

SNS_ALL = frozenset.union(SNS_NO_CMAIMED_TYPE, SNS_UPPER_LIMIT, SNS_E_LOWER_UPPER_MAGNITUDE, SNS_UNORDERED_PHOTOMETRY)


def _get_curves(sns):
    sn_files = SNFiles(sns)
    return [SNCurve.from_json(fpath) for fpath in sn_files.filepaths]


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
    def test_download_after_etag_update(self):
        pass

    def test_not_download_for_same_etag(self):
        pass


class SNFilesOfflineMode(BasicSNFilesTestCase):
    def setUp(self):
        super(SNFilesOfflineMode, self).setUp()
        sns = sorted(SNS_ALL)
        self.snnames_exist = [sns[0]]
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
        table = pandas.DataFrame(data=list(SNS_ALL), columns=['Name'])
        with NamedTemporaryFile(mode='w+', suffix='.csv') as fd:
            table.to_csv(fd.name)
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

    def read_file(self, fname):
        curve = SNCurve.from_json(fname)
        self.assertTrue(curve == curve.photometry)

    def test_download_and_read(self):
        for fname in self.sn_files.filepaths:
            self.read_file(fname)


class UpperLimitTestCase(unittest.TestCase):
    def setUp(self):
        self.curves = _get_curves(SNS_UPPER_LIMIT)

    def test_has_upper_limit(self):
        for curve in self.curves:
            has_upper_limit = reduce(operator.__or__,
                                     (np.any(lc['isupperlimit']) for lc in curve.values()))
            self.assertTrue(has_upper_limit, 'SN {} light curves should have upper limit dots')


class HasSpectraTestCase(unittest.TestCase):
    def test_has_spectra(self):
        curves = _get_curves(SNS_HAS_SPECTRA)
        for curve in curves:
            self.assertTrue(curve.has_spectra, 'SN {} data should contain spectra'.format(curve.name))

    def test_has_not_spectra(self):
        curves = _get_curves(SNS_HAS_NOT_SPECTRA)
        for curve in curves:
            self.assertFalse(curve.has_spectra, 'SN {} data should not contain spectra'.format(curve.name))


class TemporalOrderTestCase(unittest.TestCase):
    def setUp(self):
        self.sn_files = SNFiles(SNS_UNORDERED_PHOTOMETRY)

    def test_order(self):
        for fpath in self.sn_files.filepaths:
            curve = SNCurve.from_json(fpath)
            for lc in curve.values():
                self.assertTrue(np.all(np.diff(lc['time']) >= 0))

    @unittest.skipIf(six.PY2, 'Logging testing is missed in Python 2')
    def test_unordered_logging(self):
        for fpath in self.sn_files.filepaths:
            with self.assertLogs(level=logging.INFO):
                curve = SNCurve.from_json(fpath)
            del curve


class LightCurveLearningDataTestCase(unittest.TestCase):
    def setUp(self):
        with open(TEST_JSON_PATH) as fp:
            self.json_data = json.load(fp)['test']
        self.time = np.arange(0, 70, 10)
        self.u_magn = np.r_[5:-1:-3,-1:6.5:1.5]

    def test_X_U(self):
        curve = SNCurve(self.json_data, bands='U')
        assert_equal(curve.X, np.stack((np.zeros_like(self.time), self.time), axis=1))

    def test_X_states(self):
        curve = SNCurve(self.json_data)
        assert_equal(curve.X[:len(self.time),0], np.zeros_like(self.time))
        assert_equal(curve.X[len(self.time):,0], np.ones_like(self.time))

    def test_y_U(self):
        curve = SNCurve(self.json_data, bands='U')
        assert_allclose(-2.5*np.log10(curve.y*curve.y_norm), self.u_magn)
