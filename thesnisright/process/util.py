import copy
from collections import OrderedDict

from multistate_kernel.util import MultiStateData


def preprocess_curve_default(curve, bin_width=1, keep_interval=None, peak_band=None, y_err_factor=3):
    """Default light curves preprocessing pipeline

    Parameters
    ----------
    curve: SNCurve
        Data to preprocess, you should keep only necessary bands
    bin_width: float, optional
        `bin_width` used by `SNCUrve.binned`
    keep_interval: (float, float) or None, optional
        Interval around the source peak to keep. If `None`, keep all data
        points, else keep only the points between `peak + keep_interval[0]` and
        `peak + keep_interval[1]`. Good choice is
        `(-2 * approximation_range, 2 * approximation_range)`
    peak_band: str or None, optional
        This band is used to approximate a peak of the source and clamp
        observations far from the peak, see `keep_interval` for details. This
        option does nothing when `keep_interval=None`, if both `peak_band` and
        `keep_interval` are `None`, ValueError is raised
    y_err_factor: float, optional
        `err_factor` used by `SNCurve.transform_upper_limit_to_normal`

    Returns
    -------
    SNCurve

    Raises
    ------
    ValueError
    """
    curve = curve.binned(bin_width=bin_width, discrete_time=True)
    wo_ul_curve = curve.filtered(sort='filtered')
    curve = curve.transform_upper_limit_to_normal(y_factor=0, err_factor=y_err_factor)
    curve = curve.filtered(sort='filtered')

    if keep_interval is None:
        return curve
    if peak_band is None:
        raise ValueError('peak_band should be specified if keep_interval is not None')

    x_peak_approx = wo_ul_curve[peak_band].x[wo_ul_curve.odict[peak_band].y.argmax()]
    min_useful_x = x_peak_approx + keep_interval[0]
    max_useful_x = x_peak_approx + keep_interval[1]
    useful_idx = (curve.arrays.x[:, 1] >= min_useful_x) & (curve.arrays.x[:, 1] <= max_useful_x)
    msd = curve.multi_state_data().convert_arrays(curve.arrays.x[useful_idx],
                                                  curve.arrays.y[useful_idx],
                                                  curve.arrays.err[useful_idx])
    curve = curve.convert_msd(msd, is_binned=curve.is_binned, is_filtered=curve.is_filtered)
    return curve


def zero_negative_fluxes(msd, x_peak=None):
    """Convert negative y-values to zero

    Parameters
    ----------
    msd: MultiStateData
        Light curves to process
    x_peak: float or None, optional
         If `x_peak` is `None` than all negative values are zeroed. Otherwise
         the closest (in time, aka `x`) "left" and "right" negative values are
         searched and all values before "left" and after "right" time moment
         are zeroed, even if some of them are positive

    Returns
    -------
    MultistateData

    Examples
    --------
    Zero all negative values:

    >>> import numpy as np
    >>> from multistate_kernel.util import MultiStateData
    >>> from thesnisright.process.util import zero_negative_fluxes
    >>>
    >>> x = np.arange(5, dtype=np.float)
    >>> y = np.array([-3., 5., -2., 1., -77.])
    >>> err = np.zeros_like(y)  # doesn't matter in this case
    >>> lc = np.rec.fromarrays([x, y, err], names='x,y,err')
    >>> msd = MultiStateData.from_state_data({'X': lc})
    >>> zeroed_msd = zero_negative_fluxes(msd)
    >>> print(zeroed_msd.odict['X'].y)
    [0. 5. 0. 1. 0.]

    Zero "wings" around peak:

    >>> import numpy as np
    >>> from multistate_kernel.util import MultiStateData
    >>> from thesnisright.process.util import zero_negative_fluxes
    >>>
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float)
    >>> x_peak = 5.
    >>> y = np.array([1,-1,-2, 1, 2, 3, 2, 1, -1,-2,-3], dtype=np.float)
    >>> err = np.zeros_like(y)  # doesn't matter in this case
    >>> lc = np.rec.fromarrays([x, y, err], names='x,y,err')
    >>> msd = MultiStateData.from_state_data({'X': lc})
    >>> zeroed_msd = zero_negative_fluxes(msd, x_peak=x_peak)
    >>> print(zeroed_msd.odict['X'].y)
    [0. 0. 0. 1. 2. 3. 2. 1. 0. 0. 0.]

    """
    new_odict = OrderedDict()
    for band, lc in msd.odict.items():
        new_odict[band] = lc = copy.deepcopy(lc)
        if x_peak is None:
            idx = lc.y < 0
            lc.y[idx] = 0
            lc.err[idx] = 0
        else:
            for i in range(lc.x.size-1, -1, -1):
                if lc.x[i] >= x_peak:
                    continue
                if lc.y[i] <= 0:
                    i += 1
                    break
            i_left_nonzero = i
            for i in range(lc.x.size):
                if lc.x[i] <= x_peak:
                    continue
                if lc.y[i] <= 0:
                    i -= 1
                    break
            i_right_nonzero = i
            lc.y[:i_left_nonzero] = 0
            lc.err[:i_left_nonzero] = 0
            lc.y[i_right_nonzero+1:] = 0
            lc.err[i_right_nonzero+1:] = 0
    msd = MultiStateData.from_state_data(new_odict)
    return msd
