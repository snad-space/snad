#!/usr/bin/env python

import numpy as np
from sklearn.gaussian_process import kernels

from curves import OSCCurve
from interpolate import GPInterpolator, FitFailedError


def _interp(sn, plot=True, with_bazin=True):
    bands = "g',r',i'".split(',')

    k1 = kernels.RBF(length_scale_bounds=(1e-2, 1e2))
    k2 = kernels.RBF(length_scale_bounds=(1e-8, 1e8))
    # k2 = kernels.WhiteKernel()
    # k2 = kernels.ConstantKernel(constant_value_bounds='fixed')
    # k3 = kernels.ConstantKernel(constant_value_bounds='fixed')
    k3 = kernels.RBF(length_scale_bounds=(1e-8, 1e8))
    # k3 = kernels.WhiteKernel()

    m = np.array([[1, 0, 0],
                  [0.0, 1, 0],
                  [0.0, 0.0, 1]])
    m_bounds = (np.array([[1e-4, 0, 0],
                          [0, -1e3, 0],
                          [0, 0, -1e3]]),
                np.array([[1e4, 0, 0],
                          [0, 1e3, 0],
                          [0, 0, 1e3]]))

    colors = {'g': 'g', 'r': 'r', 'i': 'brown', "g'": 'g', "r'": 'r', "i'": 'brown'}

    curve = OSCCurve.from_name(sn, bands=bands, down_args={'update': False}).binned(bin_width=1, discrete_time=True)
    # curve = curve.set_error(rel=0.1)
    curve = curve.transform_upper_limit_to_normal()
    curve = curve.filtered(sort='filtered')
    x_ = np.linspace(curve.X[:, 1].min(), curve.X[:, 1].max(), 101)
    try:
        interpolator = GPInterpolator(
            curve, (k1, k2, k3), m, m_bounds,
            normalize_y=True,
            optimize_method='trust-constr',
            n_restarts_optimizer=0,
            with_bazin=with_bazin,
            raise_on_bounds=True,
            add_err=10,
            random_state=0,
            with_gp=True,
        )
    except FitFailedError:
        return sn, None
    msd = interpolator(x_, compute_err=True)
    if with_bazin:
        msd_bazin = interpolator.bazin(x_)
    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        for i, band in enumerate(bands):
            plt.subplot(2, 2, i + 1)
            plt.title(sn)
            blc = curve[band]
            plt.errorbar(blc['x'], blc['y'], blc['err'], marker='x', ls='', color=colors[band])
            plt.plot(msd.odict[band].x, msd.odict[band].y, color=colors[band], label=band)
            plt.fill_between(msd.odict[band].x,
                             msd.odict[band].y - msd.odict[band].err, msd.odict[band].y + msd.odict[band].err,
                             color='grey', alpha=0.2)
            if with_bazin:
                plt.plot(msd_bazin.odict[band].x, msd_bazin.odict[band].y, '--', color=colors[band])
            plt.ylim([0, None])
            plt.grid()
            plt.legend()
        plt.savefig('fig/{}.png'.format(sn))
    return sn, interpolator


if __name__ == '__main__':
    import os
    import pandas
    import multiprocessing

    fig_dir = 'fig'
    os.makedirs(fig_dir, exist_ok=True)

    sn_ = pandas.read_csv('gri_pr.csv').SN
    # sn_ = ['SDSS-II SN 764']

    with multiprocessing.Pool() as p:
        result = p.map(_interp, sn_[:10])
    print(result)
