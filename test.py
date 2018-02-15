#!/usr/bin/env python

import os.path
import time
from tempfile import NamedTemporaryFile
from unittest import TestCase

import pandas

from curves import SNCurve, SNFiles


SNS_TEST_LIST = [
    'SNLS-04D3fq',  # upperlimit, e_lower_magnitude/e_upper_magnitude
    'SNLS-03D3ce',  # no claimed_type
]


class DownloadTestCase(TestCase):
    def check_file(self, fpath, time_before_download):
        mod_time = os.path.getmtime(fpath)
        self.assertLessEqual(time_before_download, mod_time)

    def run_test_for_sns(self, sns):
        t = time.time()
        sn_files = SNFiles(sns, nocache=True)
        for fpath in sn_files.filepaths:
            self.check_file(fpath, t)

    def test_list_download(self):
        self.run_test_for_sns(SNS_TEST_LIST)

    def test_csv_download(self):
        table = pandas.DataFrame(data=SNS_TEST_LIST, columns=['Name'])
        with NamedTemporaryFile(mode='w+', suffix='.csv') as fd:
            table.to_csv(fd.name)
            self.run_test_for_sns(fd.name)


class ReadLightCurvesFromJsonTestCase(TestCase):
    def setUp(self):
        self.sn_files = SNFiles(SNS_TEST_LIST)

    def read_file(self, fname):
        SNCurve.from_json(fname)

    def test_download_and_read(self):
        for fname in self.sn_files.filepaths:
            self.read_file(fname)