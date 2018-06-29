import json
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from numbers import Real
from pprint import pformat

import numpy as np
from multistate_kernel.util import MultiStateData, FrozenOrderedDict
from six import iteritems, iterkeys, itervalues

try:
    from functools import lru_cache
except ImportError:
    from cachetools.func import lru_cache

from .snfiles import SNFiles


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
        return SNCurve(msd, name=self.name, is_binned=self.is_binned, is_filtered=True,
                       additional_attrs=self.__additional_attrs)

    def convert_arrays(self, x, y, err):
        return MultiStateData.from_arrays(x, y, err, self.norm, keys=self.keys())

    def convert_msd(self, msd, is_binned=False, is_filtered=False):
        """Convert MultiStateData object to SNCurve with the same attributes

        Parameters
        ----------
        msd: MultiStateData
        is_binned: bool, optional
        is_filtered: bool, optional

        Returns
        -------
        SNCurve
        """
        return SNCurve(msd, self.name, is_binned=is_binned, is_filtered=is_filtered,
                       additional_attrs=self.__additional_attrs)

    def convert_dict(self, d, is_binned=False, is_filtered=False):
        """Convert dict to SNCurve with the same attributes

        Parameters
        ----------
        d: dict-like
            It should has the same format as `.odict`
        is_binned: bool, optional
        is_filtered: bool, optional

        Returns
        -------
        SNCurve
        """
        msd = MultiStateData.from_state_data(d)
        return self.convert_msd(msd, is_binned, is_filtered)

    def multi_state_data(self):
        """Copy photometry data as MultiStateData"""
        return MultiStateData.from_state_data(self.odict)

    def msd_with_zero_valued_dots(self, x):
        """Data with dots with given x and zero y and err for all bands

        Parameters
        ----------
        x: array-like, shape=(n, )

        Returns
        -------
        MultiStateData
        """
        msd = self.multi_state_data()
        y = np.zeros_like(x)
        err = np.zeros_like(x)
        dots = {band: np.rec.array([x, y, err], names=('x', 'y', 'err')) for band in self.bands}
        msd.append_dict(dots)
        return msd

    def set_error(self, absolute=0, rel=0):
        """Return new SNCurve with set errors for dots without them

        The equation for the error is `err = absolute + rel * y`.
        Upper limits will not be changed.

        Parameters
        ----------
        absolute: float or dict-like[str: float], optional
            Can be dictionary where keys are bands
        rel: float, optional

        Returns
        -------
        SNCurve
        """
        d = dict(self.odict)
        for band, lc in d.items():
            lc = d[band] = lc.copy()
            was_writeable = lc.flags.writeable
            lc.flags.writeable = True
            inf_err_idx = (~(lc['isupperlimit'])) & (~np.isfinite(lc['err']))
            if isinstance(absolute, Real):
                abs_band = absolute
            else:
                abs_band = absolute[band]
            lc.err[inf_err_idx] = abs_band + rel * lc[inf_err_idx].y
            lc.flags.writeable = was_writeable
        msd = MultiStateData.from_state_data(d)
        return SNCurve(msd, self.name, is_binned=self.is_binned, is_filtered=False,
                       additional_attrs=self.__additional_attrs)

    def transform_upper_limit_to_normal(self, intervals=((-np.inf, None), (None, np.inf)),
                                        y_factor=0.5, err_factor=1,
                                        inf_err_is_norm=False, return_upper_limit=False):
        """New SNCurve object with upper limits converted to normal points

        Parameters
        ----------
        intervals: tuple[tuple(float or None, float or None)], optional
            A tuple of time intervals where upper limits to get. `None` on the
            first position indicates the latest normal (not upper limit)
            observation, and `None` on the last position indicates the earliest
            normal observation. The default is too take upper limits out of
            the range of normal observation data
        y_factor: float, optional
            New value of `y` is its old value multiply `y_factor`
        err_factor: float, optional
            New value of `err` is `y`'s old value multiply `err_factor`
        inf_err_is_norm: bool, optional
            Should be dot without error described as normal, when `None` is
            used in `intervals`
        return_upper_limit: bool, optional
            Return dictionary of used upper limits

        Returns
        -------
        SNCurve
        SNCurve, dict
        """
        upper_limits = OrderedDict()
        new_dict = OrderedDict()
        for band, lc in iteritems(self.odict):
            if inf_err_is_norm:
                lc_normal = lc[~lc.isupperlimit]
            else:
                lc_normal = lc[(~lc.isupperlimit) & np.isfinite(lc.err)]
            if len(lc_normal) == 0:
                t_min = np.inf
                t_max = -np.inf
            else:
                t_min = np.min(lc_normal.x)
                t_max = np.max(lc_normal.x)

            def is_between(a, interval):
                interval = list(interval)
                if interval[0] is None:
                    interval[0] = t_max
                if interval[1] is None:
                    interval[1] = t_min
                return (interval[0] < a) & (a < interval[1])

            lc = lc.copy()
            ul_idx = (reduce(lambda prev, interval: prev | is_between(lc.x, interval), intervals, False)
                      & lc.isupperlimit)
            lc.err[ul_idx] = err_factor * lc.y[ul_idx]
            lc.y[ul_idx] *= y_factor
            lc.isupperlimit[ul_idx] = False
            upper_limits[band] = lc[ul_idx]
            new_dict[band] = lc
        sn_curve = self.convert_dict(new_dict, is_binned=self.is_binned, is_filtered=self.is_filtered)
        if not return_upper_limit:
            return sn_curve
        return sn_curve, upper_limits

    def does_curve_have_rich_photometry(self, criteria=FrozenOrderedDict([('minimum', 3)])):
        """Check if curve has enough observations for futher processing

        Parameters
        ----------
        criteria: dict
            Pairs of band and minimum number of observations. Special band names:

             - `'total'` specifies minimum total number of observations, should be
               at least 1
             - `'minimum'` specifies minimum number of observations for the least
               observed band, should be at least 0
             - `'maximum'` specifies minimum number of observations for the most
               observed band, should be at least 1

        Returns
        -------
        bool
        """
        obs_dict = OrderedDict(sorted(((band, lc.x.size) for band, lc in self.odict.items()),
                                      key=lambda pair: pair[1],
                                      reverse=True))
        obs_sizes = tuple(obs_dict.values())

        criteria = dict(criteria)
        total = criteria.pop('total', 1)
        if sum(obs_sizes) < total:
            return False
        minimum = criteria.pop('minimum', 0)
        if obs_sizes[-1] < minimum:
            return False
        maximum = criteria.pop('maximum', 1)
        if obs_sizes[0] < maximum:
            return False
        for band, value in criteria.items():
            if obs_dict[band] < value:
                return False

        return True

    @property
    def bands(self):
        return tuple(self.keys())

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
        add_attrs['spectrum_count'] = 0
        if add_attrs['has_spectra']:
            add_attrs['spectrum_count'] = len(self._json['spectra'])

        if 'photometry' not in self._json:
            raise NoPhotometryError(name)
        for dot in self._json['photometry']:
            if 'time' in dot and 'band' in dot:
                # Model data, not real observation
                if 'realization' in dot or 'model' in dot:
                    continue

                # Observation of host, not target object
                if 'host' in dot:
                    continue

                if (bands is not None) and (dot.get('band') not in bands_set):
                    continue

                band_curve = d.setdefault(dot['band'], [])

                time = dot['time']
                if isinstance(time, list):
                    time = np.mean([float(t) for t in time])

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
                    time,
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
    def from_name(cls, snname, down_args=FrozenOrderedDict(), **kwargs):
        """Load photometric data by SN name, data may be downloaded

        Parameters
        ----------
        snname: string
            sne.space SN name
        down_args: dict-like
            Arguments for SNFiles
        """
        sn_files = SNFiles([snname], **down_args)
        kwargs['snname'] = snname
        return cls.from_json(sn_files.filepaths[0], **kwargs)

    @property
    def json(self):
        return deepcopy(self._json)
