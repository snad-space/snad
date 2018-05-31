import json
import logging
import os
import sys
from collections import Iterable
from copy import deepcopy
from pprint import pformat

import numpy as np
import requests
from multistate_kernel.util import MultiStateData, FrozenOrderedDict
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


def _mean_sigma(y, weight):
    if y.size == 1:
        return y[0], np.nan
    mean, sum_weight = np.average(y, weights=weight, returned=True)
    sgm_mean = np.sqrt(np.sum(weight * (y - mean) ** 2) / sum_weight / (y.size - 1))
    return mean, sgm_mean


def _typical_err(err):
    return 1 / np.sqrt(np.sum(1 / err**2))


def _mean_error(y, err):
    if y.size == 1:
        return y[0], err[0]
    weight = 1 / err**2
    mean, sgm_mean = _mean_sigma(y, weight)
    typical_err = _typical_err(err)
    if typical_err > sgm_mean:
        return mean, 0.5 * (sgm_mean + typical_err)
    return mean, sgm_mean


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


class SNCurve(MultiStateData):
    _photometry_dtype = [
        ('x', np.float),
        ('err_x', np.float),
        ('y', np.float),
        ('err', np.float),
        ('isupperlimit', np.bool)
    ]

    __doc__ = """SN photometric data.

    Represent photometric data of SN in specified bands and some additional
    metadata.

    Parameters
    ----------
    multi_state_data: MultiStateData
        Photometry data, where `x` represents time, `y` represents flux, `err`
        represents error of flux. Its `odict` attribute should contain
        `numpy.recarray` with dtype `{}`
    name: str
        SN name
    is_binned: bool
        Is initial data were binned
    is_filtered: bool
        Is initial data were filtered

    Attributes
    ----------
    name: string
        SN name.
    bands: frozenset of strings
        Photometric bands that are appeared in `photometry`.
    """.format(_photometry_dtype)

    def __init__(self, multi_state_data, name,
                 is_binned=False, is_filtered=False,
                 additional_attrs=FrozenOrderedDict()):
        for attr_name, attr_value in iteritems(additional_attrs):
            self.__setattr__(attr_name, attr_value)
        super(SNCurve, self).__init__(multi_state_data.odict, multi_state_data.arrays)
        self.name = name
        self.is_binned = is_binned
        self.is_filtered = is_filtered
        self.__additional_attrs = additional_attrs
        self.bands = self._keys_tuple

    def binned(self, bin_width, discrete_time=False, bands=None):
        """Binned photometry data

        The eges of the bins will be produced by the formula
        `time // bin_width * bin_width`. If upper limit dots are not the
        only dots in the sample, they will be excluded, as the dots with
        infinite errors. If only upper limit dots are presented, the best will
        be used, if only infinite error dots are presented, their mean will be
        used. If any dots with finite errors are presented, then weighed mean
        and corresponding error is calculated. For `discrete_time=True`
        original time errors `err_x` are ignored.

        Parameters
        ----------
        bin_width: float or None, optional
            The width of samples, in the units of photometry time dots
        discrete_time: bool, optional
            If `True`, all time steps between dots will be a multiple of
            `bin_width`, else weighted time moments will be used
        bands: iterable of str or None, optional
            Bands to use. Default is None, SNCurve.bands will be used

        Returns
        -------
        SNCurve

        Raises
        ------
        EmptyPhotometryError
        """
        if bands is None:
            bands = self.bands
        else:
            bands = _transform_to_tuple(bands)
        if set(bands).difference(self.bands):
            raise EmptyPhotometryError(self.name, set(bands) - set(self.bands))
        msd = MultiStateData.from_state_data((band, self._binning(self[band], bin_width, discrete_time))
                                             for band in bands)
        return SNCurve(msd, name=self.name,
                       is_binned=True, is_filtered=self.is_filtered,
                       additional_attrs=self.__additional_attrs)

    @staticmethod
    def _binning(blc, bin_width, discrete_time):  # blc = band light curve
        time = np.unique(blc['x'] // bin_width * bin_width)
        time_idx = np.digitize(blc['x'], time) - 1
        band_curve = np.recarray(shape=time.shape, dtype=blc.dtype)
        if discrete_time:
            band_curve['x'] = time + 0.5 * bin_width
            band_curve['err_x'] = 0.5 * bin_width
        for i, t in enumerate(time):
            sample = blc[time_idx == i]
            if np.all(sample['isupperlimit']):
                best = np.argmin(sample['y'])
                if not discrete_time:
                    band_curve['x'][i] = sample['x'][best]
                    band_curve['err_x'][i] = sample['err_x'][best]
                band_curve['y'][i] = sample['y'][best]
                band_curve['err'][i] = sample['err'][best]
                band_curve['isupperlimit'][i] = True
            elif not np.any(np.isfinite(sample['err'])):
                sample = sample[~sample['isupperlimit']]
                if not discrete_time:
                    band_curve['x'][i], band_curve['err_x'][i] = _mean_sigma(sample['x'], np.ones_like(sample['x']))
                band_curve['y'][i] = np.mean(sample['y'])
                band_curve['err'][i] = np.nan
                band_curve['isupperlimit'][i] = False
            else:
                sample = sample[np.logical_not(sample['isupperlimit']) & np.isfinite(sample['err'])]
                if not discrete_time:
                    weight = 1 / sample['err'] ** 2
                    band_curve['x'][i], band_curve['err_x'][i] = _mean_sigma(sample['x'], weight)
                band_curve['y'][i], band_curve['err'][i] = _mean_error(sample['y'], sample['err'])
                band_curve['isupperlimit'][i] = False
        return band_curve

    def filtered(self, with_upper_limits=False, with_inf_e_flux=False, bands=None, sort='default'):
        """Filtered and sorted by bands SNCurve

        Parameters
        ----------
        with_upper_limits: bool, optional
            Include observation point marked as an upper limit
        with_inf_e_flux: bool, optional
            Include observation point with infinity/NaN error
        bands: iterable of str or str or None, optional
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
        SNCurve

        Raises
        ------
        EmptyPhotometryError
        """
        if bands is None:
            bands = self.bands
        else:
            bands = _transform_to_tuple(bands)

        if (with_upper_limits
                and with_inf_e_flux
                and (bands is None or bands == self.bands)
                and sort == 'default'):
            return self

        @lru_cache(maxsize=1)
        def fd():
            if with_upper_limits and with_inf_e_flux:  # Little optimization
                return self.odict
            return {band: self[band][(np.logical_not(self[band]['isupperlimit']) + with_upper_limits)
                                     & (np.isfinite(self[band]['err']) + with_inf_e_flux)]
                    for band in bands}

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

        msd = MultiStateData.from_state_data((band, fd()[band]) for band in bands)
        if not msd.arrays.y.size:
            raise EmptyPhotometryError(self.name, bands)
        return SNCurve(msd,
                       name=self.name,
                       is_binned=self.is_binned, is_filtered=True)

    def convert_arrays(self, x, y, err):
        return MultiStateData.from_arrays(x, y, err, self.norm, keys=self.keys())

    @property
    def X(self):
        return self.arrays.x

    @property
    def y(self):
        return self.arrays.y

    @property
    def err(self):
        return self.arrays.err

    @property
    def norm(self):
        return self.arrays.norm

    def __repr__(self):
        return 'SN {}, photometry data:\n{}'.format(
            self.name, pformat(self.odict)
        )

    def __iter__(self):
        return iter(self.odict)

    def __next__(self):
        return next(self.odict)

    def __len__(self):
        return len(self.odict)

    def __getitem__(self, item):
        return self.odict[item]

    def keys(self):
        return super(SNCurve, self).keys()

    def values(self):
        return self.odict.values()

    def items(self):
        return self.odict.items()

    def iterkeys(self):
        return iterkeys(self.odict)

    def itervalues(self):
        return itervalues(self.odict)

    def iteritems(self):
        return iteritems(self.odict)


class OSCCurve(SNCurve):
    """SN photometric data from OSC JSON file

    Parameters
    ----------
    json_data: dict
        Dictionary with the data from Open Supernova Catalog json file,
        this object should contain all fields under the top-level field with
        SN name
    bands: iterable of str or str or None, optional
        Bands to use. It should be iterable of str, comma-separated str, or
        None. The default is None, all available bands will be used

    Attributes
    ----------
    name: string
        SN name.
    claimed_type: string or None
        SN claimed type, None if no claimed type is specified
    bands: frozenset of strings
        Photometric bands that are appeared in `photometry`.
    has_spectra: bool
        Is there spectral data in original JSON
    json: dict
        Original JSON data

    Raises
    ------
    NoPhotometryError
        `photometry` field is absent
    EmptyPhotometryError
        No valid photometry dots for given `bands`
    BadPhotometryDataError
        Raises if any used photometry dot contains bad data
    """
    __additional_value_fields = {
        str: ['alias', 'claimedtype', 'ra', 'dec', 'maxdate', 'maxband', 'host', 'hostra', 'hostdec'],
        float: ['redshift']
    }

    def __init__(self, json_data, bands=None):
        d = dict()

        self._json = json_data
        name = self._json['name']

        if bands is not None:
            bands = _transform_to_tuple(bands)
            bands_set = set(bands)

        add_attrs = {}
        for func, fields in iteritems(self.__additional_value_fields):
            for field in fields:
                add_attrs[field] = tuple(func(x['value']) for x in self._json.get(field, []))
        add_attrs['has_spectra'] = 'spectra' in self._json

        if 'photometry' not in self._json:
            raise NoPhotometryError(name)
        for dot in self._json['photometry']:
            if 'time' in dot and 'band' in dot:
                if (bands is not None) and (dot.get('band') not in bands_set):
                    continue

                band_curve = d.setdefault(dot['band'], [])

                if 'e_time' in dot:
                    e_time = float(dot['e_time'])
                    if e_time < 0 or not np.isfinite(e_time):
                        raise BadPhotometryDotError(name, dot, 'e_time')
                else:
                    e_time = np.nan

                magn = float(dot['magnitude'])
                flux = np.power(10, -0.4 * magn)
                if not np.isfinite(flux):
                    raise BadPhotometryDotError(name, dot)

                if 'e_lower_magnitude' in dot and 'e_upper_magnitude' in dot:
                    e_lower_magn = float(dot['e_lower_magnitude'])
                    e_upper_magn = float(dot['e_upper_magnitude'])
                    flux_lower = np.power(10, -0.4 * (magn + e_lower_magn))
                    flux_upper = np.power(10, -0.4 * (magn - e_upper_magn))
                    e_flux = 0.5 * (flux_upper - flux_lower)
                    if e_lower_magn < 0:
                        raise BadPhotometryDotError(name, dot, 'e_lower_magnitude')
                    if e_upper_magn < 0:
                        raise BadPhotometryDotError(name, dot, 'e_upper_magnitude')
                    if not np.isfinite(e_flux):
                        raise BadPhotometryDotError(name, dot)
                elif 'e_magnitude' in dot:
                    e_magn = float(dot['e_magnitude'])
                    e_flux = 0.4 * np.log(10) * flux * e_magn
                    if e_magn < 0:
                        raise BadPhotometryDotError(name, dot, 'e_magnitude')
                    if not np.isfinite(e_flux):
                        raise BadPhotometryDotError(name, dot)
                    if e_magn == 0:
                        e_flux = np.nan
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
            v = d[k] = np.rec.fromrecords(v, dtype=self._photometry_dtype)
            if np.any(np.diff(v['x']) < 0):
                logging.info('Original SN {} data for band {} contains unordered dots'.format(name, k))
                v[:] = v[np.argsort(v['x'])]
            v.flags.writeable = False

        if sum(len(v) for v in iteritems(d)) == 0:
            raise EmptyPhotometryError(name, bands)

        if bands is None:
            bands = tuple(sorted(iterkeys(d)))
        else:
            for band in bands:
                if band not in d:
                    raise EmptyPhotometryError(name, (band,))

        msd = MultiStateData.from_state_data((band, d[band]) for band in bands)
        super(OSCCurve, self).__init__(msd, name=name,
                                       is_binned=False, is_filtered=False,
                                       additional_attrs=add_attrs)

    @classmethod
    def from_json(cls, filename, snname=None, **kwargs):
        """Load photometric data from the JSON file from Open Supernova Catalog

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
    def json(self):
        return deepcopy(self._json)
