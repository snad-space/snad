#!/usr/bin/env python

import logging
import os.path
import time
import unittest
from tempfile import NamedTemporaryFile

import numpy as np
import pandas
import six

from curves import SNCurve, SNFiles


SNS_NO_CMAIMED_TYPE = {
    'SNLS-03D3ce',
}

SNS_UPPERLIMIT = {
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

SNS_ALL = set.union(SNS_NO_CMAIMED_TYPE, SNS_UPPERLIMIT, SNS_E_LOWER_UPPER_MAGNITUDE, SNS_UNORDERED_PHOTOMETRY)


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


class ReadLightCurvesFromJsonTestCase(unittest.TestCase):
    def setUp(self):
        self.sn_files = SNFiles(SNS_ALL)

    def read_file(self, fname):
        SNCurve.from_json(fname)

    def test_download_and_read(self):
        for fname in self.sn_files.filepaths:
            self.read_file(fname)


class TemporalOrderTestCase(unittest.TestCase):
    def setUp(self):
        self.sn_files = SNFiles(SNS_UNORDERED_PHOTOMETRY)

    def test_order(self):
        for fname in self.sn_files.filepaths:
            curve = SNCurve.from_json(fname)
            self.assertTrue(np.all(np.diff(curve.photometry['time']) >= 0))

    @unittest.skip(six.PY2)
    def test_unordered_logging(self):
        with self.assertLogs(level=logging.INFO):
            for fname in self.sn_files.filepaths:
                curve = SNCurve.from_json(fname)
                del curve
