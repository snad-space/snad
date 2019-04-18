import os
import unittest

from snad.load import foreign_data_samples as fds

_dir_path = os.path.dirname(__file__)
_data_path = os.path.join(_dir_path, 'data/spec_class_samples')


class BaseSNListTestCase(object):
    def test_first_sn(self):
        self.assertEqual(self.sne[0], self.first)

    def test_last_sn(self):
        self.assertEqual(self.sne[-1], self.last)

    def check_has_sn(self, sn):
        self.assertIn(sn, self.sne)

    def test_sne_in_list(self):
        for sn in self.check_list:
            self.check_has_sn(sn)


class PantheonSampleTestCase(unittest.TestCase, BaseSNListTestCase):
    def setUp(self):
        self.first = '03D1au'
        self.last = 'SCP06C0'
        self.check_list = [self.first, self.last, 'Frodo', '300179', '05D4gw']
        self.sne = fds.get_pantheon_sne_suffixes(os.path.join(_data_path, 'pantheon_sne.dat'))


class AndersonSampleTestCase(unittest.TestCase, BaseSNListTestCase):
    def setUp(self):
        self.first = 'SN1986L'
        self.last = 'SN2008in'
        self.check_list = [self.first, self.last, 'SN2006ee', 'SN0210']
        self.sne = fds.get_anderson_sne(os.path.join(_data_path, 'anderson_sne.csv'))


class SandersSampleTestCase(unittest.TestCase, BaseSNListTestCase):
    def setUp(self):
        self.first = 'PS1-10a'
        self.last = 'PS1-13esn'
        self.check_list = [self.first, self.last, 'PS1-12c', 'PS1-12auw']
        self.sne = fds.get_sanders_sne(os.path.join(_data_path, 'sanders_sne.tsv'))


class CCCPSampleTestCase(unittest.TestCase, BaseSNListTestCase):
    def setUp(self):
        self.first = 'SNF20050701-003'
        self.last = 'SN2005di'
        self.check_list = [self.first, self.last, 'SN2005az', 'Quest_SN1']
        self.sne = fds.get_cccp_sne(os.path.join(_data_path, 'cccp_sne.csv'))
