import json
import logging
import mmap
import os
import shutil
import unittest
from tempfile import mkdtemp, NamedTemporaryFile

import requests
import six
from six import BytesIO

try:
    from unittest import mock
except ImportError:
    import mock

from thesnisright.load.snfiles import SNFiles

from ._sn_lists import *


class SNFilesDifferentPathTestCase(unittest.TestCase):
    def setUp(self):
        self.parent_dir = mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.parent_dir)

    def test_dir_made(self):
        path = os.path.join(self.parent_dir, 'supernova')
        SNFiles([], path=path)
        self.assertTrue(os.path.isdir(path))


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
        with self.assertRaises(RuntimeError):
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

    @mock.patch.object(requests.Session, 'get')
    @mock.patch.object(SNFiles, '_set_file_etag')
    @mock.patch.object(SNFiles, '_get_file_etag', return_value=etag1)
    def test_download_after_etag_update(self, mock_get_file_etag, mock_set_file_etag, mock_session_get):
        response = self.response_template()
        response.headers = {'etag': self.etag2}
        mock_session_get.return_value = response

        def set_file_etag_side_effect(fpath, etag):
            self.assertEqual(etag, self.etag2)
        mock_set_file_etag.side_effect = set_file_etag_side_effect

        SNFiles(self.snnames, path=self.tmp_dir, offline=False)
        self.assertTrue(mock_get_file_etag.called)
        self.assertTrue(mock_session_get.called)
        self.assertTrue(mock_set_file_etag.called)

    @mock.patch.object(requests.Session, 'get')
    @mock.patch.object(SNFiles, '_set_file_etag')
    @mock.patch.object(SNFiles, '_get_file_etag', return_value=etag1)
    def test_not_download_for_same_etag(self, mock_get_file_etag, mock_set_file_etag, mock_session_get):
        response = self.response_template()
        response.status_code = 304
        response.iter_content = mock.Mock()
        mock_session_get.return_value = response

        SNFiles(self.snnames, path=self.tmp_dir, offline=False)
        self.assertTrue(mock_get_file_etag.called)
        self.assertTrue(mock_session_get.called)
        response.iter_content.assert_not_called()
        mock_set_file_etag.assert_not_called()

    @mock.patch.object(requests.Session, 'get')
    @mock.patch.object(SNFiles, '_get_file_etag', return_value=None)
    def test_download_if_no_xatrr(self, mock_get_file_etag, mock_session_get):
        mock_session_get.return_value = self.response_template()

        SNFiles(self.snnames, path=self.tmp_dir, offline=False)
        self.assertTrue(mock_get_file_etag.called)
        self.assertTrue(mock_session_get.called)


class SNFilesOfflineModeTestCase(BasicSNFilesTestCase):
    def setUp(self):
        super(SNFilesOfflineModeTestCase, self).setUp()
        self.snnames_exist = SNS_ALL_TUPLE[:1]
        self.snnames_not_exist = ['c4048589']
        self.sn_files_online = SNFiles(self.snnames_exist, path=self.tmp_dir, offline=False)

    def test_file_exists(self):
        sn_files = SNFiles(self.snnames_exist, path=self.tmp_dir, offline=True)
        self.check_SNfiles(sn_files)

    def test_raises_if_no_file(self):
        with self.assertRaises(ValueError):
            SNFiles(self.snnames_not_exist, path=self.tmp_dir, offline=True)


class SNFilesUpdateTestCase(BasicSNFilesTestCase):
    def setUp(self):
        super(SNFilesUpdateTestCase, self).setUp()
        self.snnames = SNS_ALL_TUPLE[:1]
        SNFiles(self.snnames, path=self.tmp_dir, offline=False)

    @mock.patch.object(SNFiles, '_download')
    def test_update_false(self, mock_download):
        SNFiles(self.snnames, path=self.tmp_dir, offline=False, update=False)
        self.assertFalse(mock_download.called)

    @mock.patch.object(SNFiles, '_download')
    def test_update_true(self, mock_download):
        SNFiles(self.snnames, path=self.tmp_dir, offline=False, update=True)
        self.assertTrue(mock_download.called)

    @mock.patch.object(SNFiles, '_download')
    def test_offline_update(self, mock_download):
        SNFiles(self.snnames, path=self.tmp_dir, offline=True, update=True)
        self.assertFalse(mock_download.called)

    @unittest.skipIf(six.PY2, 'Logging testing is missed in Python 2')
    def test_offline_update_logging(self):
        with self.assertLogs(level=logging.WARNING):
            try:
                SNFiles(self.snnames, path=self.tmp_dir, offline=True, update=True)
            except ValueError:
                pass


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
