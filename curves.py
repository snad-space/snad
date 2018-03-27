from __future__ import division

import json
import logging
import os, os.path
from collections import Iterable

import numpy as np
import requests
from six.moves import urllib
from six.moves import UserList


class SNFiles(UserList):
    """Holds a list of SNs, paths to related *.json and download these files if
    needed.
    """
    _path = 'sne/'
    _baseurl = 'https://sne.space/sne/'

    def __init__(self, sns, path=None, baseurl=None, nocache=False):
        if isinstance(sns, str):
            sns = self._names_from_csvfile(sns)
        if not isinstance(sns, Iterable):
            raise ValueError('sns should be filename or iterable')

        super(SNFiles, self).__init__(sns)

        if path is not None:
            self._path = path

        if baseurl is not None:
            self._baseurl = baseurl

        self._filenames = list( x + '.json' for x in self )
        self.filepaths = list( os.path.join(os.path.abspath(self._path), x) for x in self._filenames )
        self.urls = list( urllib.parse.urljoin(self._baseurl, urllib.parse.quote(x)) for x in self._filenames )

        self._download(nocache=nocache)

    def _download(self, nocache):
        try:
            os.makedirs(self._path)
        except OSError as e:
            if not os.path.isdir(self._path):
                raise e
        for i, filepath in enumerate(self.filepaths):
            if nocache or not os.path.exists(filepath):
                logging.info('Downloading {}'.format(self._filenames[i]))
                logging.info(self.urls[i])
                r = requests.get(self.urls[i], stream=True)
                r.raise_for_status()
                with open(filepath, 'wb') as fd:
                    for chunk in r.iter_content(chunk_size=None):
                        fd.write(chunk)

    def __repr__(self):
        return 'SN names: {sns}\nFiles: {fns}\nURLs: {urls}'.format(sns=super(SNFiles, self).__repr__(),
                                                                    fns=repr(self.filepaths),
                                                                    urls=repr(self.urls))

    @staticmethod
    def _names_from_csvfile(filename):
        """Get SN names from the `Name` column of CSV file. Such a file can
        be obtained from https://sne.space"""
        from pandas import read_csv
        data = read_csv(filename)
        return data.Name.as_matrix()


def _transform_to_set(value):
    """If `value` is a string contained comma separated spaceless sub-strings,
    these sub-strings are putted into set, other iterable will putted into set
    unmodified.
    """
    if isinstance(value, str):
        value = value.replace(' ', '')
        value = value.split(',')
    return set(value)


class BadPhotometryDataError(ValueError):
    def __init__(self, sn_name, dot, field=None):
        if field is None:
            self.message = 'SN {name} has a bad photometry item {dot}'.format(name=sn_name, dot=dot)
        else:
            self.message = 'SN {name} has a photometry item with bad {field}: {dot}'.format(name=sn_name,
                                                                                            field=field,
                                                                                            dot=dot)


class SNCurve(dict):
    __photometry_dtype = [
        ('time', np.float),
        ('e_time', np.float),
        ('flux', np.float),
        ('e_flux', np.float),
        ('isupperlimit', np.bool)
    ]
    __doc__ = """SN photometric data.

    Represent photometric data of SN in specified bands and some additional
    metadata.

    Parameters
    ----------
    photometry: dict {{str: numpy record array}}
        Photometric data in specified bands, dtype is
        `{dtype}` 
    name: string
        SN name.
    claimed_type: string or None
        SN claimed type, None if no claimed type is specified
    bands: frozenset of strings
        Photometric bands that are appeared in `photometry`.
    has_spectra: bool
        Is there spectral data in original json
    
    Raises
    ------
    BadPhotometryDataError
        Raises if any used photometry field contains bad data  
    """.format(dtype=__photometry_dtype)

    def __init__(self, json_data, bands=None):
        super(SNCurve, self).__init__()

        self._json = json_data
        self._name = self._json['name']

        if 'claimedtype' in self._json:
            self._claimed_type = self._json['claimedtype'][0]['value']  # TODO: examine other values
        else:
            self._claimed_type = None

        if bands is not None:
            bands = _transform_to_set(bands)

        self._has_spectra = 'spectra' in self._json

        self.ph = self.photometry = self
        for dot in self._json['photometry']:
            if 'time' in dot and 'band' in dot:
                if (bands is not None) and (dot.get('band') not in bands):
                    continue

                band_curve = self.setdefault(dot['band'], [])

                if 'e_time' in dot:
                    e_time = dot['e_time']
                    if e_time < 0 or not np.isfinite(e_time):
                        raise BadPhotometryDataError(self.name, dot, 'e_time')
                else:
                    e_time = np.nan

                magn = float(dot['magnitude'])
                flux = np.power(10, -0.4 * magn)
                if not np.isfinite(flux):
                    raise BadPhotometryDataError(self.name, dot)

                if 'e_lower_magnitude' in dot and 'e_upper_magnitude' in dot:
                    e_lower_magn = float(dot['e_lower_magnitude'])
                    e_upper_magn = float(dot['e_upper_magnitude'])
                    flux_lower = np.power(10, -0.4 * (magn + e_lower_magn))
                    flux_upper = np.power(10, -0.4 * (magn - e_upper_magn))
                    e_flux = 0.5 * (flux_upper - flux_lower)
                    if e_lower_magn < 0:
                        raise BadPhotometryDataError(self.name, dot, 'e_lower_magnitude')
                    if e_upper_magn < 0:
                        raise BadPhotometryDataError(self.name, dot, 'e_upper_magnitude')
                    if not np.isfinite(e_flux):
                        raise BadPhotometryDataError(self.name, dot)
                elif 'e_magnitude' in dot:
                    e_magn = float(dot['e_magnitude'])
                    e_flux = 0.4 * np.log(10) * flux * e_magn
                    if e_magn < 0:
                        raise BadPhotometryDataError(self.name, dot, 'e_magnitude')
                    if not np.isfinite(e_flux):
                        raise BadPhotometryDataError(self.name, dot)
                else:
                    e_flux = np.nan

                band_curve.append((
                    dot['time'],
                    e_time,
                    flux,
                    e_flux,
                    dot.get('upperlimit', False),
                ))
        for k, v in self.items():
            v = self[k] = np.array(v, dtype=self.__photometry_dtype)
            if np.any(np.diff(v['time']) < 0):
                logging.info('Original SN {} data for band {} contains unordered dots'.format(self._name, k))
                v[:] = v[np.argsort(v['time'])]
            v.flags.writeable = False

        self.bands = frozenset(self.keys())

    @classmethod
    def from_json(cls, filename, snname=None, **kwargs):
        """Load photometric data from json file loaded from sne.space.

        Parameters
        ----------
        filename: string
            File path.
        snname: string, optional
            Specifies a name of SN, default is automatically obtaining from
            filename or its data.
        """
        with open(filename, 'r') as fd:
            data = json.load(fd)
        if snname is None:
            snname_candidate = os.path.splitext(os.path.basename(filename))[0]
            if snname_candidate in data:
                data = data[snname_candidate]
            else:
                if len(data.keys()) == 1:
                    data = data[data.keys()[0]]
                else:
                    raise ValueError("Can't get name of SN automatically, please specify snname argument")
        else:
            data = data[snname]
        return cls(data, **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def claimed_type(self):
        return self._claimed_type

    @property
    def has_spectra(self):
        return self._has_spectra


def get_SN_curve(name):
    sn_files = SNFiles([name])
    curve = SNCurve.from_json(sn_files.filepaths[0])
    return curve
