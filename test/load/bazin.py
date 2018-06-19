import unittest
import numpy as np
from collections import OrderedDict
from multistate_kernel.util import MultiStateData
from thesnisright.interpolate.bazin import BazinFitter


class TestBazinFitter(unittest.TestCase):

    def setUp(self):
        x = np.arange(-20, 50)
        self.sigma = 0.05
        self.fall_time = 20
        self.rise_time = 5
        self.bottom = 0.2
        self.scale = 2.0

        np.random.seed(0)

        odict = OrderedDict()
        recarr = np.recarray(x.shape, [('x', 'd'), ('y', 'd'), ('err', 'd')])
        recarr['x'] = x
        recarr['y'] = self.bottom + self.scale / (np.exp(-x / self.rise_time) + np.exp(x / self.fall_time)) +\
                      np.random.normal(0, self.sigma, x.shape)
        recarr['err'] = np.repeat(self.sigma, x.shape)
        odict['r'] = recarr

        self.msd = MultiStateData.from_state_data(odict)
        self.bf = BazinFitter(self.msd, name = 'Generated MSD')
        self.bf.fit()

    def assertRelativelyEqual(self, a, b, eps):
        self.assertTrue(abs(2 * (a - b) / (a + b)) < eps, msg = '{} is not {} with eps = {}'.format(a, b, eps))

    def test_fitting(self):
        eps = 0.3
        self.assertTrue(abs(self.bf.time_shift) < 5, msg = '{} is not less {}'.format(self.bf.time_shift, 0.5))
        self.assertRelativelyEqual(self.bf.scales[0], self.scale, eps)
        self.assertRelativelyEqual(self.bf.bottoms[0], self.bottom, eps)
        self.assertRelativelyEqual(self.bf.rise_time, self.rise_time, eps)
        self.assertRelativelyEqual(self.bf.fall_time, self.fall_time, eps)

    def test_residuals(self):
        x = np.array([3.0])
        msd = self.bf(x, fill_error = True)
        self.assertTrue(msd.odict['r']['err'] > 0)
