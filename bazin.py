import curves
import numpy as np
import scipy.optimize
from collections import OrderedDict
import warnings


class InfiniteFluxErrorsError(ValueError):
    def __init__(self, name, band):
        self.message = 'Found unexpected NaN values in light curve errors band {} of {}'.format(band, name)
        super(InfiniteFluxErrorsError, self).__init__(self.message)


class NearlyEmptyFluxError(ValueError):
    def __init__(self, name, band):
        self.message = 'Found nearly empty light curve (len < 2) at band {} of {}'.format(band, name)
        super(NearlyEmptyFluxError, self).__init__(self.message)


class BazinFitter:
    """Fits the MultiStateData object to Bazin function

    Parameters
    ----------
    msd: MultiStateData
        MultiStateData to fit
    name: str
        The identification name of data. Used for warnings and errors.

    Raises
    ------
    InfiniteFluxErrorsError
    NearlyEmptyFluxError
    """

    def __init__(self, msd, name='unnamed'):
        # Store the curve
        self.curve = msd.odict
        self.bands = tuple(msd.keys())
        self.name = name

        # Check for errors
        for b in self.bands:
            if not np.all(np.isfinite(self.curve[b].err)):
                raise InfiniteFluxErrorsError(self.name, b)

            if len(self.curve[b]) < 2:
                raise NearlyEmptyFluxError(self.name, b)

        # Select the longest band and estimate the initial form parameters
        band_index = np.argmax([len(self.curve[b]) for b in self.bands])
        band = self.bands[band_index]
        (self.rise_time, self.fall_time, self.time_shift) = self._estimate_form(band)

        # Estimate the scales for all the bands
        (self.bottoms, self.scales) = self._estimate_scales()

    def _estimate_form(self, band):
        band_curve = self.curve[band]

        argmax = np.argmax(band_curve['y'])
        if argmax == 0 or argmax == len(band_curve) - 1:
            message = 'Bazin fitting to {} maybe unstable to peak absence at the band {}'.format(self.name, band)
            warnings.warn(message)

        peak_time = band_curve['x'][argmax]
        fall_time = (band_curve['x'][-1] - peak_time) / 3.0
        fall_time = np.maximum(fall_time, 2.0)
        fall_time = np.minimum(fall_time, 100.0)
        rise_time = (peak_time - band_curve['x'][0]) / 3.0
        rise_time = np.maximum(rise_time, 2.0)
        rise_time = np.minimum(rise_time, 25.0)
        time_shift = band_curve['x'][argmax] - np.log(fall_time / rise_time) / (1 / fall_time + 1 / rise_time)

        return rise_time, fall_time, time_shift

    def _estimate_scales(self):
        bands_count = len(self.bands)
        bottoms = np.empty(bands_count)
        scales = np.empty(bands_count)

        for band_index in np.arange(0, bands_count):
            band = self.bands[band_index]
            peak = np.max(self.curve[band]['y'])
            valley = np.min(self.curve[band]['y'])
            bottoms[band_index] = valley
            scales[band_index] = 2 * (peak - 0.95 * valley)

        return bottoms, scales

    def band_approximation(self, band, x):
        b = self.bands.index(band)
        return self._evaluate(x, self.rise_time,
                              self.fall_time, self.time_shift,
                              self.bottoms[b], self.scales[b])

    def __call__(self, x=None):
        odict = OrderedDict()

        xx = x
        for band in self.bands:

            if x is None:
                xx = self.curve[band]['x']

            recarr = np.recarray(len(xx), [('x', 'd'), ('y', 'd'), ('err', 'd')])
            recarr['x'] = xx
            recarr['y'] = self.band_approximation(band, xx)
            recarr['err'] = np.zeros(len(xx))
            odict[band] = recarr

        return curves.MultiStateData.from_state_data(odict)

    @staticmethod
    def _evaluate(x, rise_time, fall_time, time_shift, bottom, scale):
        form = np.exp((x - time_shift) / fall_time) + np.exp((time_shift - x) / rise_time)
        return bottom + scale / form

    @staticmethod
    def _pack_params(rise, fall, shift, bottoms, scales):
        return np.hstack((rise, fall, shift, bottoms, scales))

    def _unpack_params(self, params):
        # rise, fall, shift, bottoms, scales
        n = len(self.bottoms)
        return params[0], params[1], params[2], params[3: 3 + n], params[3 + n: 3 + 2 * n]

    def _residuals(self, params):
        (rise_time, fall_time, time_shift, bottoms, scales) = self._unpack_params(params)

        bands_number = len(bottoms)
        residuals = [None] * bands_number
        for band_index in np.arange(bands_number):
            band = self.bands[band_index]
            band_curve = self.curve[band]
            values = self._evaluate(band_curve.x, rise_time,
                                    fall_time, time_shift,
                                    bottoms[band_index], scales[band_index])
            residuals[band_index] = (values - band_curve.y) / band_curve.err

        return np.hstack(residuals)

    def fit(self):
        parameters = self._pack_params(self.rise_time, self.fall_time, self.time_shift, self.bottoms, self.scales)
        n = len(self.bottoms)
        zeros = np.zeros(n)
        infties = np.repeat(np.inf, n)
        bounds = (self._pack_params(1, 1, -np.inf, zeros, zeros),
                  self._pack_params(50, 200, +np.inf, infties, infties))
        result = scipy.optimize.least_squares(self._residuals, parameters, bounds=bounds)
        (self.rise_time, self.fall_time,
         self.time_shift, self.bottoms,
         self.scales) = self._unpack_params(result.x)


def _plot_bazin(filename, bazin):
    import matplotlib.pyplot as plt

    curve = bazin.curve

    all_band_x = np.hstack([curve[band].x for band in curve.keys()])
    xx = np.linspace(np.min(all_band_x), np.max(all_band_x))
    approx = bazin(xx)

    figure = plt.figure(figsize=(8, 10))
    for band_index in np.arange(len(bazin.bands)):
        band = bazin.bands[band_index]
        plt.subplot(len(bazin.bands), 1, band_index + 1)
        plt.plot(curve[band].x, curve[band].y, '+', approx.odict[band].x, approx.odict[band].y)
        plt.ylabel('band ' + band)

    plt.xlabel('days')

    if filename:
        plt.savefig(filename)
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    sn = 'ASASSN-14lp'
    crv = curves.OSCCurve.from_name(sn).filtered(bands=('g', 'r', 'i'))
    bf = BazinFitter(crv, name=sn)
    bf.fit()
    _plot_bazin('', bf)
