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
    flux = OrderedDict((band, np.recarray(shape=(n, ), dtype=[('x', float), ('y', float), ('err', float)]))
                       for band in new_bands)
    
    
    for i, x in enumerate(msd.odict[old_bands[0]].x):
#        print msd.odict[old_bands[0]].y
        old_flux = np.array([msd.odict[band].y[i] for band in old_bands])
        if np.any(old_flux <= 0):
            new_flux = np.full_like(b, fill_value)
        else:
            old_magn = -2.5 * np.log10(old_flux)
            
            if len(old_bands) > len(new_bands):
                vector_old_magn = old_magn - b 
    #            print 
                lstsq = np.linalg.lstsq(a, vector_old_magn)
                new_magn = lstsq[0]
    #            print lstsq
    #            print new_magn
            else:
                new_magn = np.dot(a, old_magn) + b
            new_flux = 10**(-0.4 * new_magn)

        for i_band, band in enumerate(new_bands):
            flux[band][i].x = x
            flux[band][i].y = new_flux[i_band]
            flux[band][i].err = np.nan
    return MultiStateData.from_state_data(flux)


def VR_to_gri(msd, **kwargs):
    """Convert VR light curves to gri with Lupton(2005) equation"""
    a = np.array([[1.9557, -0.9557],
                  [0.6965,  0.3035],
                  [1.7302, -0.7302]])
    b = np.array([-0.0853, 0.0688, 0.3246])
#    new_bands = ('g', 'r', 'i')
    new_bands = ("g'", "r'", "i'")
    old_bands = ('V', 'R')
    return band_transformation(msd, a, b, new_bands, old_bands=old_bands, **kwargs)


def BR_to_gri(msd, **kwargs):
    """Convert BR light curves to gri with Lupton(2005) equation"""
    a = np.array([[0.7909, 0.2091],
                  [0.1227,  0.8773],
                  [-0.2952, 1.2952]])
    b = np.array([-0.1593, 0.0573, 0.3522])
    new_bands = ('g', 'r', 'i')
#    new_bands = ("g'", "r'", "i'")
    old_bands = ('B', 'R')
    return band_transformation(msd, a, b, new_bands, old_bands=old_bands, **kwargs)

def BRI_to_gri(msd, **kwargs):
    """Convert BRI light curves to gri with Lupton(2005) equation"""
    a = np.array([[1.3130,-0.3130,0.],
                  [-0.1837,1.1837,0.],
                  [0.,0.7064,0.2936],
                  [0.,-0.2444,1.2444]])
    b = np.array([0.2271,-0.0971,-0.1439,-0.3820])
    new_bands = ('g', 'r', 'i')
#    new_bands = ("g'", "r'", "i'")
    old_bands = ('B', 'R', 'R','I')
    return band_transformation(msd, a, b, new_bands, old_bands=old_bands, **kwargs)    


if __name__ == '__main__':
    x = np.array([0.])
    err = np.array([0.])
    magnB = 20
    magnR = 20
    magnI = 20
    items = (('B', np.rec.array((x, np.array([10**(-0.4*magnB)]), err), dtype=[('x', float), ('y', float), ('err', float)])),
             ('R', np.rec.array((x, np.array([10**(-0.4*magnR)]), err), dtype=[('x', float), ('y', float), ('err', float)])),
             ('I', np.rec.array((x, np.array([10**(-0.4*magnI)]), err), dtype=[('x', float), ('y', float), ('err', float)])))
    msd = MultiStateData.from_state_data(items)
    new_msd = BRI_to_gri(msd)
    print(-2.5 * np.log10((new_msd.arrays.y * new_msd.norm)))
