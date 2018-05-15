import json
import logging
import os
import sys
from collections import Iterable, Mapping, namedtuple, OrderedDict

import numpy as np
import requests
from multistate_kernel.util import FrozenOrderedDict, data_from_items
from six import binary_type
from six import iteritems, iterkeys, itervalues
from six.moves import urllib
from six.moves import UserList

try:
    import xattr
except ImportError:
    pass

try:
    from functools import lru_cache
except ImportError:
    from cachetools.func import lru_cache


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
        from pandas import read_csv
        data = read_csv(filename)
        return data.Name.as_matrix()


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
    """.format(path=SNPaths._path, baseurl=SNPaths._baseurl)

    xattr_etag_name = b'user.etag'

    def __init__(self, sns, path=None, baseurl=None, offline=False):
        super(SNFiles, self).__init__(sns, path, baseurl)

        if not offline:
            try:
                os.makedirs(self._path)
            except OSError as e:
                if not os.path.isdir(self._path):
                    raise e

        for i, fpath in enumerate(self.filepaths):
            if offline:
                if not os.path.exists(fpath):
                    raise ValueError("Path {} should exist in offline mode".format(fpath))
            else:
                self._download(fpath, self.urls[i])

    def _download(self, fpath, url):
        etag = self._get_file_etag(fpath)

        response = self._get_response(url, etag)
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

    @staticmethod
    def _get_response(url, etag=None):
        headers = {}
        if etag is not None:
            headers['If-None-Match'] = etag
        r = requests.get(url, stream=True, headers=headers)
        r.raise_for_status()
        return r


def _transform_to_tuple(value):
    """If `value` is a string contained comma separated spaceless sub-strings,
    these sub-strings are putted into set, other iterable will putted into set
    unmodified.
    """
    if isinstance(value, str):
        value = value.replace(' ', '')
        value = value.split(',')
    return tuple(value)


class BadPhotometryDotError(ValueError):
    def __init__(self, sn_name, dot, field=None):
        if field is None:
            self.message = '{name} data file has a bad photometry item {dot}'.format(name=sn_name, dot=dot)
        else:
            self.message = '{name} data file has a photometry item with bad {field}: {dot}'.format(name=sn_name,
                                                                                                   field=field,
                                                                                                   dot=dot)
        super(BadPhotometryDotError, self).__init__(self.message)


class NoPhotometryError(ValueError):
    def __init__(self, sn_name):
        super(NoPhotometryError, self).__init__("{} data file has not field 'photometry'".format(sn_name))


class EmptyPhotometryError(ValueError):
    def __init__(self, sn_name, bands):
        if bands is None:
            self.message = '{} data file has not any photometrical observations'.format(sn_name)
        else:
            self.message = '{} data file has not any photometrical observations for bands {!s}'.format(sn_name, bands)
        super(EmptyPhotometryError, self).__init__(self.message)


class SNCurve(FrozenOrderedDict):
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
    json_data: dict
        Data from sne.space json
    bands: iterable of str or str or None, optional
        Bands to use. It should be iterable of str, comma-separated str, or
        None. The default is None, all available bands will be used
    bin_width: float or None, optional
        The width of samples, in the units of photometry time dots. The edges
        of the bins will be produced by the formula
        `time // bin_width * bin_width`. If upper limit dots are not the only
        dots in the sample, they will be excluded, as the dots with infinite
        errors. If only upper limit dots are presented, the best will be used,
        if only infinite error dots are presented, their mean will be used. If
        any dots with finite errors are presented, then weighed mean and
        corresponding error is calculated. The default is None, no sampling
        will be performed.

    Attributes
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
    NoPhotometryError
        `photometry` field is absent
    EmptyPhotometryError
        No valid photometry dots for given `bands`
    BadPhotometryDataError
        Raises if any used photometry dot contains bad data  
    """.format(dtype=__photometry_dtype)

    ScikitLearnData = namedtuple('ScikitLearnData', ('X', 'y', 'y_err', 'y_norm'))

    def __init__(self, json_data, bands=None, bin_width=None):
        d = dict()

        self._json = json_data
        self._name = self._json['name']

        if 'claimedtype' in self._json:
            self._claimed_type = self._json['claimedtype'][0]['value']  # TODO: examine other values
        else:
            self._claimed_type = None

        if bands is not None:
            bands = _transform_to_tuple(bands)
            bands_set = set(bands)

        self._has_spectra = 'spectra' in self._json

        self.ph = self.photometry = self
        if 'photometry' not in self._json:
            raise NoPhotometryError(self.name)
        for dot in self._json['photometry']:
            if 'time' in dot and 'band' in dot:
                if (bands is not None) and (dot.get('band') not in bands_set):
                    continue

                band_curve = d.setdefault(dot['band'], [])

                if 'e_time' in dot:
                    e_time = float(dot['e_time'])
                    if e_time < 0 or not np.isfinite(e_time):
                        raise BadPhotometryDotError(self.name, dot, 'e_time')
                else:
                    e_time = np.nan

                magn = float(dot['magnitude'])
                flux = np.power(10, -0.4 * magn)
                if not np.isfinite(flux):
                    raise BadPhotometryDotError(self.name, dot)

                if 'e_lower_magnitude' in dot and 'e_upper_magnitude' in dot:
                    e_lower_magn = float(dot['e_lower_magnitude'])
                    e_upper_magn = float(dot['e_upper_magnitude'])
                    flux_lower = np.power(10, -0.4 * (magn + e_lower_magn))
                    flux_upper = np.power(10, -0.4 * (magn - e_upper_magn))
                    e_flux = 0.5 * (flux_upper - flux_lower)
                    if e_lower_magn < 0:
                        raise BadPhotometryDotError(self.name, dot, 'e_lower_magnitude')
                    if e_upper_magn < 0:
                        raise BadPhotometryDotError(self.name, dot, 'e_upper_magnitude')
                    if not np.isfinite(e_flux):
                        raise BadPhotometryDotError(self.name, dot)
                elif 'e_magnitude' in dot:
                    e_magn = float(dot['e_magnitude'])
                    e_flux = 0.4 * np.log(10) * flux * e_magn
                    if e_magn < 0:
                        raise BadPhotometryDotError(self.name, dot, 'e_magnitude')
                    if not np.isfinite(e_flux):
                        raise BadPhotometryDotError(self.name, dot)
                else:
                    e_flux = np.nan

                band_curve.append((
                    dot['time'],
                    e_time,
                    flux,
                    e_flux,
                    dot.get('upperlimit', False),
                ))
        for k, v in iteritems(d):
            v = d[k] = np.array(v, dtype=self.__photometry_dtype)
            if np.any(np.diff(v['time']) < 0):
                logging.info('Original SN {} data for band {} contains unordered dots'.format(self._name, k))
                v[:] = v[np.argsort(v['time'])]

            if bin_width is not None:
                v = d[k] = self._binning(v, bin_width)

            v.flags.writeable = False

        if sum(len(v) for v in iteritems(d)) == 0:
            raise EmptyPhotometryError(self.name, bands)

        if bands is None:
            bands = tuple(sorted(iterkeys(d)))
        else:
            for band in bands:
                if band not in d:
                    raise EmptyPhotometryError(self.name, (band,))
        self.bands = bands

        super(SNCurve, self).__init__(((band, d[band]) for band in self.bands))

    @staticmethod
    def _binning(blc, bin_width):  # blc = band light curve
        time = np.unique(blc['time'] // bin_width * bin_width)
        time_idx = np.digitize(blc['time'], time) - 1
        band_curve = np.empty(shape=time.shape, dtype=blc.dtype)
        band_curve['time'] = time + 0.5 * bin_width
        band_curve['e_time'] = 0.5 * bin_width
        for i, t in enumerate(time):
            sample = blc[time_idx == i]
            if np.all(sample['isupperlimit']):
                best = np.argmin(sample['flux'])
                band_curve['flux'][i] = sample['flux'][best]
                band_curve['e_flux'][i] = sample['e_flux'][best]
                band_curve['isupperlimit'][i] = True
            elif not np.any(np.isfinite(sample['e_flux'])):
                sample = sample[~sample['isupperlimit']]
                band_curve['flux'][i] = np.mean(sample['flux'])
                band_curve['e_flux'][i] = np.nan
                band_curve['isupperlimit'][i] = False
            else:
                sample = sample[np.logical_not(sample['isupperlimit']) & np.isfinite(sample['e_flux'])]
                weight = 1 / sample['e_flux'] ** 2
                sum_weight = np.sum(weight)
                flux = np.sum(weight * sample['flux']) / sum_weight
                if sample.size == 1:
                    e_flux = sample['e_flux'][0]
                else:
                    sgm_mean = np.sqrt(np.sum(weight * (sample['flux'] - flux) ** 2) / sum_weight / (sample.size - 1))
                    typical_e_flux = 1 / np.sqrt(sum_weight)
                    if typical_e_flux > sgm_mean:
                        e_flux = 0.5 * (sgm_mean + typical_e_flux)
                    else:
                        e_flux = sgm_mean
                band_curve['flux'][i] = flux
                band_curve['e_flux'][i] = e_flux
                band_curve['isupperlimit'][i] = False
        return band_curve

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

    @classmethod
    def from_name(cls, snname, path=None, **kwargs):
        """Load photometric data by SN name, data may be downloaded

        Parameters
        ----------
        snname: string
            sne.space SN name
        path: string or None, optional
            Specifies local path of json data, for default see `SNFiles`
        """
        sn_files = SNFiles([snname], path=path)
        kwargs['snname'] = snname
        return cls.from_json(sn_files.filepaths[0], **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def claimed_type(self):
        return self._claimed_type

    @property
    def has_spectra(self):
        return self._has_spectra

    def multi_state_data(self, with_upper_limits=False, with_inf_e_flux=False, bands=None, sort='default'):
        """Filtered and sorted by bands MultiStateData object

        Parameters
        ----------
        with_upper_limits: bool, optional
            Include observation point marked as an upper limit
        with_inf_e_flux: bool, optional
            Include observation point with infinity/NaN error
        bands: iterable or None, optional
            Bands to return. Default is None, SNCurves.bands will be used
        sort: str, optional
            How `bands` will be sorted. Should be one of the following
            strings:

                - 'default' will keep the order of `bands`
                - 'alphabetic' or 'alpha' will sort `bands` alphabetically
                - 'total' will sort `bands` by the total number of photometric
                  points, from maximum to minimum
                - 'filtered' will sort `bands` by the number of returned
                  photometric points from maximum to minimum, e.g. points
                  filtered by `with_upper_limits` and `with_inf_e_flux`
                  arguments

        Returns
        -------
        MultiStateData

        Raises
        ------
        EmptyPhotometryError
        """
        @lru_cache(maxsize=1)
        def fd():
            if with_upper_limits and with_inf_e_flux:  # Little optimization
                return self
            return {band: self[band][(np.logical_not(self[band]['isupperlimit']) + with_upper_limits)
                                      & (np.isfinite(self[band]['e_flux']) + with_inf_e_flux)]
                    for band in bands}

        if bands is None:
            bands = self.bands
        else:
            bands = _transform_to_tuple(bands)

        if sort == 'default':
            pass
        elif sort == 'alphabetic' or sort == 'alpha':
            bands = sorted(bands)
        elif sort == 'total':
            bands = sorted(bands, key=lambda band: self[band].size, reverse=True)
        elif sort == 'filtered':
            bands = sorted(bands, key=lambda band: fd()[band].size, reverse=True)
        else:
            raise ValueError('Argument sort={} is not supported'.format(sort))

        def items_generator():
            for band in bands:
                yield (band, (fd()[band][name] for name in ('time', 'flux', 'e_flux')))
        ph = items_generator()
        msd = data_from_items(ph)
        if not msd.arrays.y.size:
            raise EmptyPhotometryError(self.name, bands)
        return msd

    @property
    @lru_cache(maxsize=1)
    def _Xy(self):
        return self.multi_state_data(with_upper_limits=False,
                                     with_inf_e_flux=False,
                                     bands=None,
                                     sort='filtered').arrays

    @property
    def X(self):
        return self._Xy.x

    @property
    def y(self):
        return self._Xy.y

    @property
    def y_err(self):
        return self._Xy.y_err

    @property
    def y_norm(self):
        return self._Xy.norm

    def __repr__(self):
        return 'SN {} with claimed type {}. Photometry data:\n{}'.format(
            self.name, self.claimed_type, repr(self._d)
        )
