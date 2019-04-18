import unittest

import numpy as np
from numpy.testing import assert_allclose
from multistate_kernel.util import MultiStateData

from snad import transform_bands_msd, BRI_to_gri


class BRI2griTestCase(unittest.TestCase):
    def test_msd(self):
        x = np.array([0.])
        err = np.array([0.])
        magnB = 20
        magnR = 20
        magnI = 20
        items = (('B', np.rec.array((x, np.array([10 ** (-0.4 * magnB)]), err),
                                    dtype=[('x', float), ('y', float), ('err', float)])),
                 ('R', np.rec.array((x, np.array([10 ** (-0.4 * magnR)]), err),
                                    dtype=[('x', float), ('y', float), ('err', float)])),
                 ('I', np.rec.array((x, np.array([10 ** (-0.4 * magnI)]), err),
                                    dtype=[('x', float), ('y', float), ('err', float)])))
        msd = MultiStateData.from_state_data(items)
        new_msd = transform_bands_msd(msd, BRI_to_gri)
        magn = -2.5 * np.log10((new_msd.arrays.y * new_msd.norm))
        assert_allclose(magn, [19.84211267, 20.06126321, 20.32025792])
