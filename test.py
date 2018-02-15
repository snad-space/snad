#!/usr/bin/env python

from unittest import TestCase

from curves import SNCurve, SNFiles


class DownloadAndRead(TestCase):
    def setUp(self):
        self.sns = ['SNLS-04D3fq', 'SN2004S', 'SN2005W']

    def read_file(self, fname):
        SNCurve.from_json(fname)

    def test_download_and_read(self):
        sn_files = SNFiles(self.sns)
        for fname in sn_files.filepaths:
            self.read_file(fname)