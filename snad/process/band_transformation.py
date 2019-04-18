from collections import OrderedDict
from functools import reduce
from operator import eq

import numpy as np
from six import itervalues, iteritems
from multistate_kernel.util import MultiStateData


class BandTransformation:
    """Transforms light curves from one band set to another

    Parameters
    ----------
    light_curve: dict or array-like
        Initial light curves. Keys are band names, values are arrays of
        magnitudes. All values should have the same shape
    a: 2-D numpy.array, shape=(len(new_bands), len(old_bands)))
    b: 1-D numpy.array, shape=(len(new_bands), )
    new_bands: iterable of str
        A collection of new band names
    old_bands: iterable of str
        A collection of old band names

    Returns
    -------
    dict[str: numpy.array]

    """

    def __init__(self, a, b, old_bands, new_bands):
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.old_bands = old_bands
        self.new_bands = new_bands
        if len(old_bands) > len(new_bands):
            self._new_magn = self._lstsq
        else:
            self._new_magn = self._linear

    def _lstsq(self, old_magn):
        return np.linalg.lstsq(self.a, old_magn - self.b, rcond=None)[0]

    def _linear(self, old_magn):
        return np.dot(self.a, old_magn) + self.b

    def __call__(self, light_curve):
        old = {k: np.asarray(v) for k, v in iteritems(light_curve)}
        shape = old[next(iter(old))].shape
        if not reduce(eq, (shape == v.shape for v in itervalues(old))):
            raise ValueError('All initial values should have the same shape')
        new = {k: np.empty(shape) for k in self.new_bands}

        for idx in np.ndindex(shape):
            old_magn = np.array([old[k][idx] for k in self.old_bands])
            new_magn = self._new_magn(old_magn)
            for i, k in enumerate(self.new_bands):
                new[k][idx] = new_magn[i]
        return new


VR_to_gri = BandTransformation(
    [[1.9557, -0.9557],
     [0.6965,  0.3035],
     [1.7302, -0.7302]],
    [-0.0853, 0.0688, 0.3246],
    ('V', 'R',),
    ("g'", "r'", "i'",)
)

BR_to_gri = BandTransformation(
    [[0.7909, 0.2091],
     [0.1227,  0.8773],
     [-0.2952, 1.2952]],
    [-0.1593, 0.0573, 0.3522],
    ('B', 'R',),
    ('g', 'r', 'i',),
)

BRI_to_gri = BandTransformation(
    [[1.3130,-0.3130,0.],
     [-0.1837,1.1837,0.],
     [0.,0.7064,0.2936],
     [0.,-0.2444,1.2444]],
    [0.2271,-0.0971,-0.1439,-0.3820],
    ('B', 'R', 'R', 'I',),  # two 'R' is not an error, see shape of a
    ('g', 'r', 'i'),
)


def transform_bands_msd(msd, transform, fill_value=np.nan):
    """Convert light curve to another band set

    Parameters
    ----------
    msd: MultiStateData
        Initial light curve object, `y` is `10^(-0.4 magn)` as produced by
        `OSCCurve`. All light curves should have the same length
    transform: BandTransformation
        One of `VR_to_gri`, `BR_to_gri`, `BRI_to_gri`, defined in this module
    fill_value: float
        Value to fill corrupt data, i.e. non-positive fluxes

    Returns
    -------
    MultiStateData

    """
    old_dict = {band: -2.5 * np.log10(msd.odict[band].y) for band in transform.old_bands}
    magn = transform(old_dict)
    new_odict = OrderedDict()
    x = msd.odict[transform.old_bands[0]].x
    err = np.full_like(x, np.nan)
    for band in transform.new_bands:
        y = 10**(-0.4 * magn[band])
        y[np.isnan(y)] = fill_value
        new_odict[band] = np.rec.array((x, y, err), dtype=[('x', float), ('y', float), ('err', float)])
    new_msd = MultiStateData.from_state_data(new_odict)
    return new_msd


__all__ = ('VR_to_gri', 'BR_to_gri', 'BRI_to_gri', 'transform_bands_msd',)
