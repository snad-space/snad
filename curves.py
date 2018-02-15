from __future__ import division

import json
import logging
import os, os.path
from collections import Iterable

import numpy as np
import requests
import six
from six.moves import urllib
from six.moves import UserList


if six.PY2:
    BAND_DTYPE = 'S8'
    SN_TYPE_DTYPE = 'S8'
else:
    BAND_DTYPE = 'U8'
    SN_TYPE_DTYPE = 'U8'


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
        return 'SN names: {sns}\nFiles: {fns}\nURLs: {urls}'.format(sns=super().__repr__(),
                                                                    fns=repr(self.filepaths),
                                                                    urls=repr(self.urls))

    @staticmethod
    def _names_from_csvfile(filename):
        """Get SN names from the `Name` column of CSV file. Such the file can
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


class SNCurve():
    __photometry_dtype = [
        ('time', np.float),
        ('e_time', np.float),
        ('flux', np.float),
        ('e_flux', np.float),
        ('band', BAND_DTYPE),
    ]
    __doc__ = """SN photometric data.

    Represent photometric data of SN in specified bands and some additional
    metadata.

    Parameters
    ----------
    photometry: numpy record array
        Photometric data in specified bands, dtype is
        `{dtype}` 
    name: string
        SN name.
    claimed_type: string or None
        SN claimed type, None if no claimed type is specified
    bands: set of strings
        Photometric bands that are appeared in `photometry`.   
    """.format(dtype=__photometry_dtype)

    def __init__(self, json_data, bands=None):
        self._data = json_data
        self._name = self._data['name']

        if 'claimedtype' in self._data:
            self._claimed_type = self._data['claimedtype'][0]['value']  # TODO: examine other values
        else:
            self._claimed_type = None

        if bands is not None:
            bands = _transform_to_set(bands)

        def photometry_generator():
            for dot in self._data['photometry']:
                if 'time' in dot and 'band' in dot and not dot.get('upperlimit', False):
                    if (bands is not None) and (dot.get('band') not in bands):
                        continue
                    magn = float(dot['magnitude'])
                    flux = np.power(10, -0.4*magn)
                    if 'e_lower_magnitude' in dot and 'e_upper_magnitude' in dot:
                        flux_lower = np.power(10, -0.4*(magn+float(dot['e_lower_magnitude'])))
                        flux_upper = np.power(10, -0.4*(magn-float(dot['e_upper_magnitude'])))
                        e_flux = 0.5 * (flux_upper - flux_lower)
                    elif 'e_magnitude' in dot:
                        e_flux = 0.4 * np.log(10) * flux * float(dot['e_magnitude'])
                    else:
                        e_flux = np.nan
                    yield (
                        dot['time'],
                        dot.get('e_time', np.nan),
                        flux,
                        e_flux,
                        dot['band'],
                    )

        self.photometry = np.fromiter(photometry_generator(), dtype=self.__photometry_dtype)
        # All photometry dates should be sorted, so it is cheaper to check it than sort every time:
        if np.any(np.diff(self.photometry['time']) < 0):
            logging.info('Original SN {} data contains unordered dots'.format(self._name))
            self.photometry[:] = self.photometry[np.argsort(self.photometry['time'])]
        self.photometry.flags.writeable = False

        self.bands = frozenset(self.photometry['band'])

    # def spline(self, band=None, delta_mag=0.01, k=3):
    #     if band is None:
    #         if len(self.bands) > 1:
    #             raise ValueError('Please, specify band, because len(self.bands) > 1')
    #         else:
    #             band = self.bands.copy().pop()
    #     photo_in_band = self.photometry[self.photometry['band']==band]
    #
    #     if photo_in_band.shape[0] < k+1:
    #         return np.zeros_like
    #
    #     epsilon_flux = 1. - 10.**(-0.4 * delta_mag)
    #     flux = np.power(10., -0.4 * photo_in_band['mag'])
    #     weight = 1. / (epsilon_flux * flux)
    #     time = photo_in_band['time'] - photo_in_band['time'][flux.argmax()]
    #     # spline = interpolate.interp1d(time, flux,
    #     #                               copy=False,
    #     #                               bounds_error=False, fill_value=0,
    #     #                               kind='cubic')
    #     test_time = np.average(time)
    #     max_flux_gradient = None
    #     s = time.shape[0] / 2.
    #     while True:
    #         s *= 2
    #         spline = interpolate.UnivariateSpline(time, flux, w=weight, s=s, k=k, ext='zeros')
    #         # Check the smoothing spline is self-constistent
    #         if spline.get_residual() > s:
    #             continue
    #         # Check if one value of the spline is NaN
    #         if np.isnan(spline(test_time)):
    #             continue
    #         # Check if any values in interpolated dots are NaN
    #         if np.any(np.isnan(spline(time))):
    #             continue
    #         derivatives = spline.derivative()(time)
    #         # Check if any derivatives in interpolated dots are NaN
    #         if np.any(np.isnan(derivatives)):
    #             continue
    #         if max_flux_gradient is None:
    #             flux_gradient = np.gradient(flux, np.gradient(time))
    #             flux_gradient[np.isinf(flux_gradient)] = 0
    #             max_flux_gradient = np.max(np.abs(flux_gradient))
    #         # Check if any derivative is too steep
    #         if np.any(np.abs(derivatives[k-1:1-k]) > max_flux_gradient):
    #             continue
    #         break
    #
    #     return spline

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

    def __repr__(self):
        return repr(self.photometry)

    @property
    def name(self):
        return self._name

    @property
    def claimed_type(self):
        return self._claimed_type


# class SNDataForLearning(SNFiles):
#     def __init__(self, *args, bands=None, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.curves = []
#         for i, filepath in enumerate(self.filepaths):
#             self.curves.append(SNCurve.from_json(filepath, snname=self[i], bands=bands))
#
#         if bands is None:
#             self.bands = sorted(set(chain.from_iterable(curve.photometry['band'] for curve in self.curves)))
#         else:
#             self.bands = bands
#
#     def X_y(self, n_dots=100, time_interval=(-50, 350), normalize_flux=True):
#         self.n_dots = n_dots
#         self.time_interval = time_interval
#
#         self.time_dots = np.linspace(*self.time_interval, self.n_dots)
#
#         self.X = np.empty((len(self.curves), n_dots * len(self.bands)), dtype=np.float)
#         self.y = np.fromiter( [curve.claimedtype for curve in self.curves], dtype=SN_TYPE_DTYPE, count=len(self.curves) )
#         for i_curve, curve in enumerate(self.curves):
#             for i_band, band in enumerate(self.bands):
#                 dots = curve.spline(band)(self.time_dots)
#                 if normalize_flux:
#                     dots /= dots.max()
#                 self.X[i_curve, i_band * n_dots:(i_band + 1) * n_dots] = dots
#
#         return self.X, self.y

