#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from multistate_kernel.kernel import MultiStateKernel
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from curves import SNCurve


LINE_STYLES = ("-","--","-.",":")


if __name__ == '__main__':
    curve = SNCurve.from_name('SNLS-04D3fq', bands='i,r,z')
    curve_i = SNCurve.from_name('SNLS-04D3fq', bands='i')
    curve_r = SNCurve.from_name('SNLS-04D3fq', bands='r')
    curve_z = SNCurve.from_name('SNLS-04D3fq', bands='z')

    rbf1 = kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    rbf2 = kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    rbf3 = kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    white1 = kernels.WhiteKernel(noise_level=1.0, noise_level_bounds='fixed')
    white2 = kernels.WhiteKernel(noise_level=1.0, noise_level_bounds='fixed')
    msk = MultiStateKernel(
        (rbf1, rbf2, white1),
        np.array([[1,0.5,0.],[0.5,1,0.],[0.5,0.5,1]]),
        [np.array([[1e-2,-10,-10],[-10,1e-2,-10],[-10,-10,-100]]), np.array([[1e2,10,10],[10,1e2,10],[10,10,1e2]])]
    )

    t_ = np.r_[curve.X[:,1].min():curve.X[:,1].max():101j]
    x_ = np.block([[np.full_like(t_,0), np.full_like(t_, 1), np.full_like(t_,2)], [t_, t_, t_]]).T

    gpr1 = GaussianProcessRegressor(1*rbf1, alpha=curve_i.y_err**2)
    gpr2 = GaussianProcessRegressor(1*rbf1, alpha=curve_r.y_err**2)
    gpr3 = GaussianProcessRegressor(1*rbf1, alpha=curve_z.y_err**2)
    gpr1.fit(curve_i.X[:,1].reshape(-1,1), curve_i.y)
    gpr2.fit(curve_r.X[:,1].reshape(-1,1), curve_r.y)
    # gpr3.fit(curve_z.X[:,1].reshape(-1,1), curve_z.y)
    y1_ = gpr1.predict(t_.reshape(-1,1))
    y2_ = gpr2.predict(t_.reshape(-1,1))
    # y3_ = gpr3.predict(t_.reshape(-1,1))
    print(gpr1.kernel_.get_params())
    print(gpr2.kernel_.get_params())
    # print(gpr3.kernel_.get_params())

    gpr = GaussianProcessRegressor(msk, alpha=curve.y_err**2)
    gpr.fit(curve.X, curve.y)
    y_, y_err_ = gpr.predict(x_, return_std=True)
    y_samples_ = gpr.sample_y(x_, n_samples=3)
    print(gpr.kernel_.get_params())
    print('Log Likelihood = ', gpr.log_marginal_likelihood_value_)

    plt.errorbar(curve_i.X[:,1], curve_i.y*curve_i.y_norm, curve_i.y_err*curve_i.y_norm, color='b', fmt='x', label='observation - i')
    plt.errorbar(curve_r.X[:,1], curve_r.y*curve_r.y_norm, curve_r.y_err*curve_r.y_norm, color='r', fmt='x', label='observation - r')
    plt.errorbar(curve_z.X[:,1], curve_z.y*curve_z.y_norm, curve_z.y_err*curve_z.y_norm, color='g', fmt='x', label='observation - z')

    plt.plot(t_, y1_*curve_i.y_norm, '--b', label='individual - i')
    plt.plot(t_, y2_*curve_r.y_norm, '--r', label='individual - r')
    # plt.plot(t_, y3_*curve_z.y_norm, '--g', label='individual - z')

    plt.plot(t_, y_[:len(t_)]*curve.y_norm, 'b', label='correlated - i')
    plt.fill_between(t_, (y_[:len(t_)]-y_err_[:len(t_)])*curve.y_norm, (y_[:len(t_)]+y_err_[:len(t_)])*curve.y_norm, color='b', alpha=0.1)
    plt.plot(t_, y_[len(t_):2*(len(t_))]*curve.y_norm, 'r', label='correlated - r')
    plt.fill_between(t_, (y_[len(t_):2*len(t_)]-y_err_[len(t_):2*len(t_)])*curve.y_norm, (y_[len(t_):2*len(t_)]+y_err_[len(t_):2*len(t_)])*curve.y_norm, color='r', alpha=0.1)
    plt.plot(t_, y_[2*len(t_):]*curve.y_norm, 'g', label='correlated - z')
    plt.fill_between(t_, (y_[2*len(t_):]-y_err_[2*len(t_):])*curve.y_norm, (y_[2*len(t_):]+y_err_[2*len(t_):])*curve.y_norm, color='g', alpha=0.1)

    for i, sample in enumerate(y_samples_.T):
        ls = LINE_STYLES[i+1]
        plt.plot(t_, sample[:len(t_)]*curve.y_norm, 'b', lw=0.5, ls=ls)
        plt.plot(t_, sample[len(t_):2*len(t_)]*curve.y_norm, 'r', lw=0.5, ls=ls)
        plt.plot(t_, sample[2*len(t_):]*curve.y_norm, 'g', lw=0.5, ls=ls)

    plt.legend()
    plt.savefig('multistate.pdf')
    plt.show()
