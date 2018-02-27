#!/usr/bin/env python

import logging
import operator
import os.path
import time
import unittest
from functools import reduce
from tempfile import NamedTemporaryFile

import numpy as np
import pandas
import six

from curves import SNCurve, SNFiles


SNS_NO_CMAIMED_TYPE = {
    'SNLS-03D3ce',
}

SNS_UPPER_LIMIT = {
    'SNLS-04D3fq',
    'PS1-10ahf',
}

SNS_E_LOWER_UPPER_MAGNITUDE = {
    'SNLS-04D3fq',
}

SNS_UNORDERED_PHOTOMETRY = {
    'PTF09atu',
    'PS1-10ahf',
}

SNS_HAS_SPECTRA = {
    'SNLS-04D3fq',
}

SNS_HAS_NOT_SPECTRA = {
    'SN1993A',
}

SNS_ALL = set.union(SNS_NO_CMAIMED_TYPE, SNS_UPPER_LIMIT, SNS_E_LOWER_UPPER_MAGNITUDE, SNS_UNORDERED_PHOTOMETRY)


def _get_curves(sns):
    sn_files = SNFiles(sns)
    return [SNCurve.from_json(fpath) for fpath in sn_files.filepaths]


class DownloadTestCase(unittest.TestCase):
    def check_file(self, fpath, time_before_download):
        mod_time = os.path.getmtime(fpath)
        self.assertLessEqual(time_before_download, mod_time)

    def run_test_for_sns(self, sns):
        t = time.time()
        sn_files = SNFiles(sns, nocache=True)
        for fpath in sn_files.filepaths:
            self.check_file(fpath, t)

    def test_list_download(self):
        self.run_test_for_sns(SNS_ALL)

    def test_csv_download(self):
        table = pandas.DataFrame(data=list(SNS_ALL), columns=['Name'])
        with NamedTemporaryFile(mode='w+', suffix='.csv') as fd:
            table.to_csv(fd.name)
            self.run_test_for_sns(fd.name)


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
