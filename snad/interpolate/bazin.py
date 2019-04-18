import numpy as np
import scipy.optimize
import scipy.linalg
from collections import OrderedDict
import warnings
from multistate_kernel.util import MultiStateData


class InfiniteFluxErrorsError(ValueError):
    def __init__(self, name, band):
        self.message = 'Found unexpected NaN values in light curve errors band {} of {}'.format(band, name)
        super(InfiniteFluxErrorsError, self).__init__(self.message)


class NearlyEmptyFluxError(ValueError):
    def __init__(self, name, band):
        self.message = 'Found nearly empty light curve (len < 2) at band {} of {}'.format(band, name)
        super(NearlyEmptyFluxError, self).__init__(self.message)


class NoFitPerformedError(ValueError):
    def __init__(self, name):
        self.message = 'Some method relying on fitting results called before fit itself for {}'.format(name)
        super(NoFitPerformedError, self).__init__(self.message)


class FitCovarianceMissingError(ValueError):
    def __init__(self, name):
        self.message = 'Missing covariance data requested for {}. '.format(name) +\
                       'Possible reasons are misfit or fit at the bounds.'
        super(FitCovarianceMissingError, self).__init__(self.message)


class BazinFitter(object):
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

        # Places to store the results of fitting and precalculated covariance matrix
        self.result = None
        self.covariance = None

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
        """Evaluates the approximation at the specific band."""

        b = self.bands.index(band)
        return self._evaluate(x, self.rise_time,
                              self.fall_time, self.time_shift,
                              self.bottoms[b], self.scales[b])

    def band_approximation_error(self, band, x):
        """Estimates the errors of approximation at the specific band."""

        if self.covariance is None:
            if self.result is None:
                raise NoFitPerformedError(self.name)
            else:
                raise FitCovarianceMissingError(self.name)

        bands_number = len(self.bands)
        band_index = self.bands.index(band)
        params_index = np.arange(5)
        params_index[3] = 3 + band_index  # Don't act like that, kids.
        params_index[4] = 3 + bands_number + band_index  # That's not a good code really.
        covariance = self.covariance[np.ix_(params_index, params_index)]

        grads = self._evaluate_gradient(x, self.rise_time,
                                        self.fall_time, self.time_shift,
                                        self.bottoms[band_index], self.scales[band_index])
        errors = np.empty(x.shape)
        for index in np.arange(len(errors)):
            g = grads[index]
            errors[index] = np.sqrt(np.sum(g.T.dot(covariance) * g.T))

        return errors

    def __call__(self, x=None, fill_error=False):
        """Evaluates the approximation for all the bands.

        Parameters
        ----------
        x: 1-d ndarray or None
            Either evaluate all the bands at the specific x values, or use initial
            ones when the supplied x is equal to None.
        fill_error: bool
            Should the approximation errors be estimated.
        """

        odict = OrderedDict()

        xx = x
        for band in self.bands:

            if x is None:
                xx = self.curve[band]['x']

            recarr = np.recarray(len(xx), [('x', 'd'), ('y', 'd'), ('err', 'd')])
            recarr['x'] = xx
            recarr['y'] = self.band_approximation(band, xx)
            recarr['err'] = self.band_approximation_error(band, xx) if fill_error else np.zeros(len(xx))
            odict[band] = recarr

        return MultiStateData.from_state_data(odict)

    @staticmethod
    def _evaluate(x, rise_time, fall_time, time_shift, bottom, scale):
        form = np.exp((x - time_shift) / fall_time) + np.exp((time_shift - x) / rise_time)
        return bottom + scale / form

    @staticmethod
    def _evaluate_gradient(x, rise_time, fall_time, time_shift, bottom, scale):
        form = np.exp((x - time_shift) / fall_time) + np.exp((time_shift - x) / rise_time)
        common_mul = scale / form ** 2
        grad = np.empty((len(x), 5))
        grad[:, 0] = common_mul * np.exp((time_shift - x) / rise_time) * (time_shift - x) / rise_time ** 2
        grad[:, 1] = common_mul * np.exp((x - time_shift) / fall_time) * (x - time_shift) / fall_time ** 2
        grad[:, 2] = common_mul * (np.exp((x - time_shift) / fall_time) / fall_time -
                                   np.exp((time_shift - x) / rise_time) / rise_time)
        grad[:, 3] = 1
        grad[:, 4] = 1 / form
        return grad

    @staticmethod
    def _pack_params(rise, fall, shift, bottoms, scales):
        return np.hstack((rise, fall, shift, bottoms, scales))

    def _unpack_params(self, params):
        # rise, fall, shift, bottoms, scales
        n = len(self.bottoms)
        return params[0], params[1], params[2], params[3: 3 + n], params[3 + n: 3 + 2 * n] # Some more Fortran

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

    def _residuals_gradient(self, params):
        (rise_time, fall_time, time_shift, bottoms, scales) = self._unpack_params(params)

        bands_number = len(self.bands)
        grads = [None] * bands_number
        params_index = np.arange(5)
        for band_index in np.arange(bands_number):
            band = self.bands[band_index]
            band_curve = self.curve[band]
            params_index[3] = 3 + band_index # Ahhh, sweet Fortran77.
            params_index[4] = 3 + bands_number + band_index # Who the hell know how to do this right?
            grads[band_index] = np.zeros((len(self.curve[band]), 3 + 2 * bands_number))
            band_grad = self._evaluate_gradient(band_curve.x, rise_time,
                                                fall_time, time_shift,
                                                bottoms[band_index], scales[band_index])
            band_grad /= np.reshape(band_curve.err, (-1, 1))
            grads[band_index][:, params_index] = band_grad

        return np.vstack(grads)

    def fit(self, use_gradient=True):
        """Do the main procedure. Fit the initial parameter values to the previousely supplied data."""

        parameters = self._pack_params(self.rise_time, self.fall_time, self.time_shift, self.bottoms, self.scales)
        bands_number = len(self.bottoms)

        # Calculate the scales for the least squares
        bottom_scales = np.empty(bands_number)
        scales_scales = np.empty(bands_number)
        times = [None] * bands_number
        for band_index in np.arange(bands_number):
            band = self.bands[band_index]
            bottom_scales[band_index] = np.min(self.curve[band].y)
            scales_scales[band_index] = np.max(self.curve[band].y) - bottom_scales[band_index]
            times[band_index] = self.curve[band].x

        time_scale = np.mean(np.hstack(times))
        p_scales = self._pack_params(10.0, 20.0, time_scale, bottom_scales, scales_scales)

        # Set the boundaries
        zeros = np.zeros(bands_number)
        infties = np.repeat(np.inf, bands_number)
        bounds = (self._pack_params(1, 1, -np.inf, zeros, zeros),
                  self._pack_params(50, 200, +np.inf, infties, infties))

        optimargs = {}
        if use_gradient:
            optimargs['jac'] = self._residuals_gradient

        result = scipy.optimize.least_squares(self._residuals, parameters,
                                              bounds=bounds, x_scale=p_scales, **optimargs)
        self.result = result

        (self.rise_time, self.fall_time,
         self.time_shift, self.bottoms,
         self.scales) = self._unpack_params(result.x)

        # Try to calculate covariance matrix
        if np.all(result.active_mask == 0) and result.fun.size > result.x.size:
            # The method (and code) is driven from scipy.optimize.curve_fit
            _, s, vt = scipy.linalg.svd(result.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
            non_zero_values = s > threshold
            s = s[non_zero_values]
            vt = vt[non_zero_values]
            covariance = np.dot(vt.T / s ** 2, vt)
            sigma_sq = 2 * result.cost / (result.fun.size - result.x.size)
            covariance *= sigma_sq
            self.covariance = covariance
        else:
            self.covariance = None

        return 2 * result.cost, result.fun * np.hstack(self.curve[band].err for band in self.bands)


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
    from snad import OSCCurve

    sn = 'ASASSN-14lp'
    crv = OSCCurve.from_name(sn).filtered(bands=('g', 'r', 'i'))
    bf = BazinFitter(crv, name=sn)
    bf.fit()
    _plot_bazin('', bf)
