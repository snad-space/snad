#!/usr/bin/env python3

from __future__ import print_function, division

import json
import logging
import os, os.path
import numpy as np
import requests
import sys
import urllib.parse
from collections import Iterable
from itertools import chain
from scipy import interpolate


BAND_DTYPE = 'U8'
SN_TYPE_DTYPE = 'U8'
MAX_INT = sys.maxsize


class SNFiles(list):
    _path = 'sne/'
    _baseurl = 'https://sne.space/sne/'

    def __init__(self, sns, path=None, baseurl=None, nocache=False):
        if isinstance(sns, str):
            sns = self._names_from_csvfile(sns)
        if not isinstance(sns, Iterable):
            raise ValueError('sns should be filename or iterable')

        super().__init__(sns)

        if path is not None:
            self._path = path

        if baseurl is not None:
            self._baseurl = baseurl

        self._filenames = list( x + '.json' for x in self )
        self.filepaths = list( os.path.join(os.path.abspath(self._path), x) for x in self._filenames )
        self.urls = list( urllib.parse.urljoin(self._baseurl, urllib.parse.quote(x)) for x in self._filenames )

        self._download(nocache=nocache)

    def _download(self, nocache):
        os.makedirs(self._path, exist_ok=True)
        for i, filepath in enumerate(self.filepaths):
            if nocache or not os.path.exists(filepath):
                logging.info('Downloading {}'.format(self._filenames[i]))
                logging.info(self.urls[i])
                r = requests.get(self.urls[i], stream=True)
                r.raise_for_status()
                with open(filepath, 'wb') as fd:
                    for chunk in r.iter_content(chunk_size=4096):
                        fd.write(chunk)

    def __repr__(self):
        return 'SN names: {sns}\nFiles: {fns}\nURLs: {urls}'.format(sns=super().__repr__(),
                                                                    fns=repr(self.filepaths),
                                                                    urls=repr(self.urls))

    @staticmethod
    def _names_from_csvfile(filename):
        from pandas import read_csv
        data = read_csv(filename)
        return data.Name.as_matrix()


class ContainsEverythingExceptNone:
    def __contains__(self, item):
        return item is not None

    def __len__(self):
        return MAX_INT


def transform_value_to_set_like(value):
    if value is None:
        return ContainsEverythingExceptNone()
    else:
        # if isinstance(value, str):
        #    value = value.encode()
        if isinstance(value, str):
            value = value.split(',')
        return set(value)


class SNCurve:
    def __init__(self, json_data, bands=None):
        self._data = json_data
        self.bands = transform_value_to_set_like(bands)
        self.name = self._data['name']
        self.claimedtype = self._data['claimedtype'][0]['value']
        self.photometry = self._construct_photometry()

    def _construct_photometry(self):
        dots = []
        for dot in self._data['photometry']:
            if dot.get('band') in self.bands and 'time' in dot:
                dots.append(( dot['time'], dot['magnitude'], dot['band'] ))
        return np.array(dots,
                        dtype=[('time', np.float), ('mag', np.float), ('band', BAND_DTYPE)])

    def spline(self, band=None, delta_mag=0.01, k=3):
        if band is None:
            if len(self.bands) > 1:
                raise('Please, specify band, because len(self.bands) > 1')
            else:
                band = self.bands.copy().pop()
        photo_in_band = self.photometry[self.photometry['band']==band]

        if photo_in_band.shape[0] < k+1:
            return np.zeros_like

        epsilon_flux = 1. - 10.**(-0.4 * delta_mag)
        flux = np.power(10., -0.4 * photo_in_band['mag'])
        weight = 1. / (epsilon_flux * flux)
        time = photo_in_band['time'] - photo_in_band['time'][flux.argmax()]
        # spline = interpolate.interp1d(time, flux,
        #                               copy=False,
        #                               bounds_error=False, fill_value=0,
        #                               kind='cubic')
        test_time = np.average(time)
        max_flux_gradient = None
        s = time.shape[0] / 2.
        while True:
            s *= 2
            spline = interpolate.UnivariateSpline(time, flux, w=weight, s=s, k=k, ext='zeros')
            # Check the smoothing spline is self-constistent
            if spline.get_residual() > s:
                continue
            # Check if one value of the spline is NaN
            if np.isnan(spline(test_time)):
                continue
            # Check if any values in interpolated dots are NaN
            if np.any(np.isnan(spline(time))):
                continue
            derivatives = spline.derivative()(time)
            # Check if any derivatives in interpolated dots are NaN
            if np.any(np.isnan(derivatives)):
                continue
            if max_flux_gradient is None:
                flux_gradient = np.gradient(flux, np.gradient(time))
                flux_gradient[np.isinf(flux_gradient)] = 0
                max_flux_gradient = np.max(np.abs(flux_gradient))
            # Check if any derivative is too steep
            if np.any(np.abs(derivatives[k-1:1-k]) > max_flux_gradient):
                continue
            break

        return spline

    @classmethod
    def from_json(cls, filename, snname=None, **kwargs):
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


class SNDataForLearning(SNFiles):
    def __init__(self, *args, bands=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.curves = []
        for i, filepath in enumerate(self.filepaths):
            self.curves.append(SNCurve.from_json(filepath, snname=self[i], bands=bands))

        if bands is None:
            self.bands = sorted(set(chain.from_iterable(curve.photometry['band'] for curve in self.curves)))
        else:
            self.bands = bands

    def X_y(self, n_dots=100, time_interval=(-50, 350), normalize_flux=True):
        self.n_dots = n_dots
        self.time_interval = time_interval

        self.time_dots = np.linspace(*self.time_interval, self.n_dots)

        self.X = np.empty((len(self.curves), n_dots * len(self.bands)), dtype=np.float)
        self.y = np.fromiter( [curve.claimedtype for curve in self.curves], dtype=SN_TYPE_DTYPE, count=len(self.curves) )
        for i_curve, curve in enumerate(self.curves):
            for i_band, band in enumerate(self.bands):
                dots = curve.spline(band)(self.time_dots)
                if normalize_flux:
                    dots /= dots.max()
                self.X[i_curve, i_band * n_dots:(i_band + 1) * n_dots] = dots

        return self.X, self.y


################################################################################


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # c = SNFiles('/Users/hombit/Downloads/Unknown.csv')
    # logging.info(c)
    #
    # s = SNCurve.from_json('sne/SN1991T.json', bands=None)
    # logging.info(s)
    # s.spline('V')

    d = SNDataForLearning('./SN_phot100.csv', bands='V')
    X, y = d.X_y(n_dots=100, time_interval=(-50, 350))
