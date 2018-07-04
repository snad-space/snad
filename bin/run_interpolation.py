#!/usr/bin/env python

import os

import numpy as np
from sklearn.gaussian_process import kernels

from thesnisright import OSCCurve, BazinFitter, SNFiles
from thesnisright.interpolate.gp import GPInterpolator


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SNE_PATH = os.path.join(PROJECT_ROOT, 'sne')
COLORS = {'g': 'g', 'r': 'r', 'i': 'brown', "g'": 'g', "r'": 'r', "i'": 'brown'}


def _interp(sn, bands, peak_band, rng=np.linspace(-50, 100, 151), fig_dir='.', plot=True, with_bazin=True):
    orig_curve = OSCCurve.from_json(sn, bands=bands)
    curve = orig_curve.binned(bin_width=1, discrete_time=True)
    # curve = curve.set_error(rel=0.1)
    curve = curve.transform_upper_limit_to_normal(y_factor=1e-3*with_bazin, err_factor=3)
    curve = curve.filtered(sort='filtered')
    min_length = [max(1, np.max(np.diff(lc.x))) for lc in curve.odict.values()]

    k1 = kernels.RBF(length_scale_bounds=(min_length[0], 1e4))
    k2 = kernels.RBF(length_scale_bounds=(min_length[1], 1e4))
    # k2 = kernels.WhiteKernel()
    # k2 = kernels.ConstantKernel(constant_value_bounds='fixed')
    # k3 = kernels.ConstantKernel(constant_value_bounds='fixed')
    k3 = kernels.RBF(length_scale_bounds=(min_length[2], 1e4))
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

    if with_bazin:
        bazin_fitter = BazinFitter(curve, curve.name)
        bazin_fitter.fit(use_gradient=True)
        curve = curve - bazin_fitter()
    interpolator = GPInterpolator(
        curve, (k1, k2, k3), m, m_bounds,
        normalize_y=with_bazin,
        optimize_method='trust-constr',
        n_restarts_optimizer=0,
        raise_on_bounds=False,
        add_err=0,
        random_state=0,
    )

    x_for_peak_search = np.linspace(curve.X[:, 1].min(), curve.X[:, 1].max(), 101)
    msd_for_peak = interpolator(x_for_peak_search, compute_err=False)
    if with_bazin:
        msd_bazin_for_peak = bazin_fitter(x_for_peak_search)
        msd_for_peak = msd_bazin_for_peak + msd_bazin_for_peak
    x_peak = msd_for_peak.odict[peak_band].x[np.argmax(msd_for_peak.odict[peak_band].y)]
    x_ = rng + x_peak

    msd = interpolator(x_, compute_err=True)
    if with_bazin:
        msd_bazin = bazin_fitter(x_)
        msd_plus_bazin = msd + msd_bazin

    if plot:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.clf()
        if with_bazin:
            fig, ax_ = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
        else:
            fig, ax_ = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
            ax_ = np.atleast_2d(ax_)
        for i, band in enumerate(bands):
            plt.sca(ax_[0, i])
            if i == 1:
                plt.title('{}  {}{}'.format(
                    orig_curve.name,
                    ', '.join(curve.keys()),
                    ' Near bounds'*interpolator.near_bounds
                ))

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
            # plt.ylim([0, None])
            plt.grid()
            plt.legend()

            if with_bazin:
                plt.sca(ax_[1, i])
                odict = curve.odict[band]
                plt.errorbar(odict.x, odict.y, odict.err, marker='x', ls='', color=COLORS[band])
                plt.plot(msd.odict[band].x, msd.odict[band].y, color=COLORS[band], label=band)
                plt.fill_between(msd.odict[band].x,
                                 msd.odict[band].y - msd.odict[band].err, msd.odict[band].y + msd.odict[band].err,
                                 color='grey', alpha=0.2)
                plt.grid()
                plt.legend()
        fig_path = os.path.join(fig_dir, '{}.png'.format(orig_curve.name))
        print(fig_path)
        plt.savefig(fig_path)
    return sn, interpolator


if __name__ == '__main__':
    import os
    import multiprocessing
    from functools import partial

    fig_dir = os.path.join(PROJECT_ROOT, 'fig')
    os.makedirs(fig_dir, exist_ok=True)

    sn_files = SNFiles(os.path.join(PROJECT_ROOT, 'data/min3obs_g,r,i.csv'),
                       update=False, path=SNE_PATH)
    interp = partial(_interp, bands=("g", "r", "i"), peak_band="r", fig_dir=fig_dir, plot=True, with_bazin=False)
    with multiprocessing.Pool() as p:
        result = p.map(interp, sn_files.filepaths)
    print(result)
