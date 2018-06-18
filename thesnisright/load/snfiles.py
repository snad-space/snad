import logging
import os
import sys
from collections import Iterable

import requests
from six import binary_type
from six.moves import urllib
from six.moves import UserList

try:
    import xattr
except ImportError:
    pass


class SNPaths(UserList):
    _path = 'sne/'
    _baseurl = 'https://sne.space/sne/'

    def __init__(self, sns, path, baseurl):
        if isinstance(sns, str):
            sns = self._names_from_csvfile(sns)
        if not isinstance(sns, Iterable):
            raise ValueError('sns should be filename or iterable')

        super(SNPaths, self).__init__(sns)

        if path is None:
            path = self._path
        if baseurl is None:
            baseurl = self._baseurl
        self.path = path
        self.baseurl = baseurl

        self._filenames = list( x + '.json' for x in self )
        self.filepaths = list( os.path.join(os.path.abspath(self.path), x) for x in self._filenames )
        self.urls = list(
            urllib.parse.urljoin(self.baseurl, urllib.parse.quote(x))
            for x in self._filenames
        )

    def __repr__(self):
        return 'SN names: {sns}\nFiles: {fns}\nURLs: {urls}'.format(sns=super(SNPaths, self).__repr__(),
                                                                    fns=repr(self.filepaths),
                                                                    urls=repr(self.urls))

    @staticmethod
    def _names_from_csvfile(filename):
        """Get SN names from the `Name` column of CSV file. Such a file can
        be obtained from https://sne.space"""
        from csv import reader
        with open(filename) as fd:
            csv_reader = reader(fd)
            fields = next(csv_reader)
            i = fields.index('Name')
            sn_names = [row[i] for row in csv_reader]
        return sn_names


class SNFiles(SNPaths):
    """Holds a list of SNs, paths to related *.json and download these files if
    needed.

    Parameters
    ----------
    sns: list of strings
        Names of SNs, i.e. name of target json file without extension
    path: string, optional
        Path to local directory to hold *.json files. If None, path
        {path} will be used
    baseurl: string, optional
        First part of URL to ask for SN json data. If None, URL
        {baseurl} will be used
    offline: bool
        No new data will be downloaded. ValueError will be raised if target
        file cannot be found.
    update: bool
        Check if new data is available and download it. Useful only when
        `offline=False`
    """.format(path=SNPaths._path, baseurl=SNPaths._baseurl)

    xattr_etag_name = b'user.etag'

    def __init__(self, sns, path=None, baseurl=None, offline=False, update=True):
        super(SNFiles, self).__init__(sns, path, baseurl)

        if offline and update:
            logging.warning('SNFiles.__init__: it is worthless to specify both offline=True and update=True')

        if not offline:
            try:
                os.makedirs(self._path)
            except OSError as e:
                if not os.path.isdir(self._path):
                    raise e

        if offline:
            for fpath in self.filepaths:
                if not os.path.isfile(fpath):
                    raise ValueError("Path {} should exist in offline mode".format(fpath))
            return

        with requests.session() as session:
            for i, fpath in enumerate(self.filepaths):
                if update or not os.path.exists(fpath):
                    self._download(fpath, self.urls[i], session=session)

    def _download(self, fpath, url, session=requests):
        etag = self._get_file_etag(fpath)
        headers = {}
        if etag is not None:
            headers['If-None-Match'] = etag

        with session.get(url, stream=True, headers=headers) as response:
            if response.status_code == requests.codes.not_modified:
                logging.info('File {} is up to data, skip downloading'.format(fpath))
                return
            elif response.status_code != requests.codes.ok:
                raise RuntimeError('HTTP status code should be 200 or 304, not {}'.format(response.status_code))

            logging.info('Downloading {} to {}'.format(url, fpath))
            with open(fpath, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=4096):
                    fd.write(chunk)

        if 'etag' in response.headers:
            self._set_file_etag(fpath, response.headers['etag'])

    def _get_file_etag(self, fpath):
        if not os.path.exists(fpath) or 'xattr' not in sys.modules:
            return None
        try:
            return xattr.getxattr(fpath, self.xattr_etag_name)
        except OSError:
            return None

    def _set_file_etag(self, fpath, etag):
        if 'xattr' not in sys.modules:
            return
        if not isinstance(etag, binary_type):
            etag = etag.encode('utf-8')
        xattr.setxattr(fpath, self.xattr_etag_name, etag)

