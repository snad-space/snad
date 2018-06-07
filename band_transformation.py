from collections import OrderedDict

import numpy as np
from multistate_kernel.util import MultiStateData


def band_transformation(msd, a, b, new_bands, old_bands=None, fill_value=np.nan):
    """Transforms light curves from one band set to another

    Parameters
    ----------
    msd: MultiStateData
        Initial light curves
    a: 2-D numpy.array, shape=(len(new_bands), len(old_bands)))
    b: 1-D numpy.array, shape=(len(new_bands), )
    new_bands: iterable of str
        A collection of new band names
    old_bands: iterable of str or None, optional
        A collection of old band names. If None, `msd.keys()` will be used
    fill_value: float, optional
        The value to fill output data when input flux is negative

    Returns
    -------
    MultiStateData
    """
    if old_bands is None:
        old_bands = tuple(msd.keys())
    n = len(msd.arrays.y) // len(msd.keys())
    flux = OrderedDict((band, np.recarray(shape=(n, ), dtype=msd.odict[old_bands[0]].dtype)) for band in new_bands)
    for i, x in enumerate(msd.odict[old_bands[0]].x):
        old_flux = np.array([msd.odict[band][i].y for band in old_bands])
        if np.any(old_flux <= 0):
            new_flux = np.full_like(b, fill_value)
        else:
            old_magn = -2.5 * np.log10(old_flux)
            new_magn = np.dot(a, old_magn) + b
            new_flux = 10**(-0.4 * new_magn)
        for i_band, band in enumerate(new_bands):
            flux[band][i].x = x
            flux[band][i].y = new_flux[i_band]
            flux[band][i].err = np.nan
    return MultiStateData.from_state_data(flux)


def VR_to_gri(msd, **kwargs):
    """Convert VR light curves to gri"""
    a = np.array([[1.9557, -0.9557],
                  [0.6965,  0.3035],
                  [1.7302, -0.7302]])
    b = np.array([-0.0853, 0.0688, 0.3246])
    new_bands = ('g', 'r', 'i')
    old_bands = ('V', 'R')
    return band_transformation(msd, a, b, new_bands, old_bands=old_bands, **kwargs)


if __name__ == '__main__':
    x = np.array([0.])
    err = np.array([0.])
    magnV = 1
    magnR = 2
    items = (('V', np.rec.array((x, np.array([10**(-0.4*magnV)]), err), dtype=[('x', float), ('y', float), ('err', float)])),
             ('R', np.rec.array((x, np.array([10**(-0.4*magnR)]), err), dtype=[('x', float), ('y', float), ('err', float)])),)
    msd = MultiStateData.from_state_data(items)
    new_msd = VR_to_gri(msd)
    print(-2.5 * np.log10((new_msd.arrays.y * new_msd.norm)))
