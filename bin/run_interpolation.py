#!/usr/bin/env python

import numpy as np
from sklearn.gaussian_process import kernels

from thesnisright import OSCCurve, BazinFitter
from thesnisright.interpolate.gp import GPInterpolator, FitFailedError


COLORS = {'g': 'g', 'r': 'r', 'i': 'brown', "g'": 'g', "r'": 'r', "i'": 'brown'}


def _interp(sn, fig_dir='.', plot=True, with_bazin=True):
    bands = "g',r',i'".split(',')

    k1 = kernels.RBF(length_scale_bounds=(3, 1e4))
    k2 = kernels.RBF(length_scale_bounds=(3, 1e4))
    # k2 = kernels.WhiteKernel()
    # k2 = kernels.ConstantKernel(constant_value_bounds='fixed')
    # k3 = kernels.ConstantKernel(constant_value_bounds='fixed')
    k3 = kernels.RBF(length_scale_bounds=(3, 1e4))
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

    orig_curve = OSCCurve.from_name(sn, bands=bands, down_args={'update': False})
    curve = orig_curve.binned(bin_width=1, discrete_time=True)
    # curve = curve.set_error(rel=0.1)
    curve = curve.transform_upper_limit_to_normal()
    curve = curve.filtered(sort='filtered')
    x_ = np.linspace(curve.X[:, 1].min()-20, curve.X[:, 1].max()+20, 101)

    if with_bazin:
        bazin_fitter = BazinFitter(curve, curve.name)
        bazin_fitter.fit(use_gradient=True)
        curve = curve - bazin_fitter()
        msd_bazin = bazin_fitter(x_)
    interpolator = GPInterpolator(
        curve, (k1, k2, k3), m, m_bounds,
        normalize_y=with_bazin,
        optimize_method='trust-constr',
        n_restarts_optimizer=0,
        raise_on_bounds=False,
        add_err=0,
        random_state=0,
    )
    msd = interpolator(x_, compute_err=True)
    if with_bazin:
        msd_plus_bazin = msd + msd_bazin

    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        if with_bazin:
            vsize = 3
            plt.figure(figsize=(10, 6))
        else:
            vsize = 2
            plt.figure()
        for i, band in enumerate(bands):
            plt.subplot(2, vsize, i + 1)
            if i == 1:
                plt.title('{}  {}{}'.format(sn, ', '.join(curve.keys()), ' Near bounds'*interpolator.near_bounds))

            blc = orig_curve.odict[band]
            normal = blc[(np.isfinite(blc.err)) & (~blc.isupperlimit)]
            wo_errors = blc[(~np.isfinite(blc.err)) & (~blc.isupperlimit)]
            upper_limit = blc[blc.isupperlimit]
            plt.errorbar(normal.x, normal.y, normal.err, marker='x', ls='', label=band, color=COLORS[band])
            plt.plot(wo_errors.x, wo_errors.y, '*', color=COLORS[band])
            plt.plot(upper_limit.x, upper_limit.y, 'v', color=COLORS[band])

            if with_bazin:
                plt.plot(msd_bazin.odict[band].x, msd_bazin.odict[band].y, '--', color=COLORS[band])
                odict = msd_plus_bazin.odict
            else:
                odict = msd.odict
            plt.plot(odict[band].x, odict[band].y, color=COLORS[band])
            plt.fill_between(odict[band].x,
                             odict[band].y - odict[band].err, odict[band].y + odict[band].err,
                             color='grey', alpha=0.2)
            plt.ylim([0, None])
            plt.grid()
            plt.legend()

            if with_bazin:
                plt.subplot(2, vsize, i + 4)
                odict = curve.odict[band]
                plt.errorbar(odict.x, odict.y, odict.err, marker='x', ls='', color=COLORS[band])
                plt.plot(msd.odict[band].x, msd.odict[band].y, color=COLORS[band], label=band)
                plt.fill_between(msd.odict[band].x,
                                 msd.odict[band].y - msd.odict[band].err, msd.odict[band].y + msd.odict[band].err,
                                 color='grey', alpha=0.2)
                plt.grid()
                plt.legend()
        plt.savefig(os.path.join(fig_dir, '{}.png'.format(sn)))
    return sn, interpolator


if __name__ == '__main__':
    import os
    import pandas
    import multiprocessing
    from functools import partial

    fig_dir = '../fig'
    os.makedirs(fig_dir, exist_ok=True)

    sn_ = sorted(pandas.read_csv('../data/gri_pr.csv').SN)
    # _interp('SN2007sk')

    interp = partial(_interp, fig_dir=fig_dir, plot=True, with_bazin=True)
    with multiprocessing.Pool() as p:
        result = p.map(interp, sn_)
    print(result)
