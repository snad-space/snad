#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from multistate_kernel.kernel import MultiStateKernel
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from curves import get_SN_curve


if __name__ == '__main__':
    curve = get_SN_curve('SNLS-04D3fq')
    i = curve['i'][np.logical_not(curve['i']['isupperlimit'])]
    r = curve['r'][np.logical_not(curve['r']['isupperlimit'])]
    x_data = np.block([[np.zeros_like(i['time']), np.ones_like(r['time'])],
                       [i['time'], r['time']]]).T
    y_data = np.r_[i['flux'], r['flux']]
    y_norm = y_data.max()
    y_data /= y_norm
    sigma = np.r_[i['e_flux'], r['e_flux']] / y_norm

    rbf1 = kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    rbf2 = kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    # white = kernels.WhiteKernel(noise_level=1.0, noise_level_bounds='fixed')
    msk = MultiStateKernel((rbf1, rbf2), np.array([[1,0.5],[0.5,1]]), [np.array([[1e-2,-1.0],[-1.0,1e-2]]), np.array([[1e2,1.0],[1.0,1e2]])])

    t_ = np.r_[x_data[:,1].min():x_data[:,1].max():101j]
    x_ = np.block([[np.zeros_like(t_), np.ones_like(t_)], [t_, t_]]).T

    gpr1 = GaussianProcessRegressor(rbf1, alpha=sigma[:len(i)]**2)
    gpr2 = GaussianProcessRegressor(rbf1, alpha=sigma[len(i):]**2)
    gpr1.fit(x_data[:len(i),1,np.newaxis], y_data[:len(i)])
    gpr2.fit(x_data[len(i):,1,np.newaxis], y_data[len(i):])
    y1_ = gpr1.predict(t_.reshape(-1,1))
    y2_ = gpr2.predict(t_.reshape(-1,1))
    print(gpr1.kernel_.get_params())
    print(gpr2.kernel_.get_params())


    gpr = GaussianProcessRegressor(msk, alpha=sigma**2)
    gpr.fit(x_data, y_data)
    y_ = gpr.predict(x_)
    print(gpr.kernel_.get_params())

    plt.plot(i['time'], i['flux']/y_norm, 'xb', label='observation - i')
    plt.plot(r['time'], r['flux']/y_norm, 'xr', label='observation - r')

    plt.plot(t_, y1_, '--b', label='individual - i')
    plt.plot(t_, y2_, '--r', label='individual - r')

    plt.plot(t_, y_[:len(t_)], 'b', label='correlated - i')
    plt.plot(t_, y_[len(t_):], 'r', label='correlated - r')

    plt.legend()
    plt.show()