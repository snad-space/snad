#!/usr/bin/env python

import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from multistate_kernel.util import MultiStateData

from snad import OSCCurve, BazinFitter, SNFiles, transform_bands_msd
from snad.interpolate.gp import GPInterpolator
from snad.process.util import preprocess_curve_default, zero_negative_fluxes


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
FIG_ROOT = os.path.join(PROJECT_ROOT, 'fig')
SNE_PATH = os.path.join(PROJECT_ROOT, 'sne')
COLORS = {'B': 'b', 'R': 'deeppink', 'I': 'olive',
          'g': 'g', 'r': 'r', 'i': 'brown',
          "g'": 'g', "r'": 'r', "i'": 'brown',}


def _peak(interpolator, peak_band, bazin_fitter=None):
    X = interpolator.msd.X
    x_for_peak_search = np.linspace(X[:, 1].min(), X[:, 1].max(), 101)
    msd = interpolator(x_for_peak_search, compute_err=False)
    if bazin_fitter is not None:
        msd_bazin = bazin_fitter(x_for_peak_search)
        msd = msd + msd_bazin
    x_peak = msd.odict[peak_band].x[np.argmax(msd.odict[peak_band].y)]
    return x_peak


def _plot(fig_dir, orig_curve, interpolator, msd, msd_bazin=None, msd_plus_bazin=None, msd_transformed=None):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.clf()
    vsize = 1 + int(msd_bazin is not None) + int(msd_transformed is not None)
    fig, ax_ = plt.subplots(vsize, 3, figsize=(10, 4*vsize), sharex=True)
    ax_ = np.atleast_2d(ax_)
    for i, band in enumerate(bands):
        plt.sca(ax_[0, i])
        if i == 1:
            plt.title('{}  {}{}'.format(
                orig_curve.name,
                ', '.join(msd.odict.keys()),
                ' Near bounds' * interpolator.near_bounds
            ))

        blc = orig_curve.odict[band]
        normal = blc[(np.isfinite(blc.err)) & (~blc.isupperlimit)]
        wo_errors = blc[(~np.isfinite(blc.err)) & (~blc.isupperlimit)]
        upper_limit = blc[blc.isupperlimit]
        plt.errorbar(normal.x, normal.y, normal.err, marker='x', ls='', label=band, color=COLORS[band])
        plt.plot(wo_errors.x, wo_errors.y, '*', color=COLORS[band])
        plt.plot(upper_limit.x, upper_limit.y, 'v', color=COLORS[band])

        if msd_bazin is not None and msd_plus_bazin is not None:
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

        if msd_bazin is not None:
            plt.sca(ax_[1, i])
            odict = msd.odict[band]
            plt.errorbar(odict.x, odict.y, odict.err, marker='x', ls='', color=COLORS[band])
            plt.plot(msd.odict[band].x, msd.odict[band].y, color=COLORS[band], label=band)
            plt.fill_between(msd.odict[band].x,
                             msd.odict[band].y - msd.odict[band].err, msd.odict[band].y + msd.odict[band].err,
                             color='grey', alpha=0.2)
            plt.grid()
            plt.legend()

    if msd_transformed is not None:
        for i, (band, lc) in enumerate(msd_transformed.odict.items()):
            plt.sca(ax_[vsize-1, i])
            plt.plot(lc.x, lc.y, color=COLORS[band], label=band)
            plt.grid()
            plt.legend()

    fig_path = os.path.join(fig_dir, '{}.png'.format(orig_curve.name))
    print(fig_path)
    plt.savefig(fig_path)


def interpolate(sn, bands, peak_band, rng=np.linspace(-50, 100, 151), fig_dir='.',
                plot=True, with_bazin=True, transform=None):
    rng_width = rng.max() - rng.min()

    orig_curve = OSCCurve.from_json(sn, bands=bands)
    curve = preprocess_curve_default(orig_curve, keep_interval=(-2*rng_width, 2*rng_width), peak_band=peak_band)

    kernels_ = GPInterpolator.default_initial_kernels(curve, min_length_limits=(1, 0.5*rng_width))
    m, m_bounds = GPInterpolator.default_constant_matrix(len(curve))
    bazin_fitter = None
    if with_bazin:
        bazin_fitter = BazinFitter(curve, curve.name)
        bazin_fitter.fit(use_gradient=True)
        curve = curve - bazin_fitter()
    interpolator = GPInterpolator(
        curve, kernels_, m, m_bounds,
        normalize_y=with_bazin,
        optimize_method='trust-constr',
        n_restarts_optimizer=0,
        raise_on_bounds=False,
        add_err=0,
        random_state=0,
    )

    x_peak = _peak(interpolator, peak_band, bazin_fitter)
    x_ = rng + x_peak

    msd = interpolator(x_, compute_err=True)
    msd_bazin = None
    msd_plus_bazin = None
    if with_bazin:
        msd_bazin = bazin_fitter(x_)
        msd_plus_bazin = msd + msd_bazin

    msd = zero_negative_fluxes(msd, x_peak)

    msd_transformed = None
    if transform is not None:
        if with_bazin:
            msd_transformed = transform_bands_msd(msd_plus_bazin, transform, fill_value=0)
        else:
            msd_transformed = transform_bands_msd(msd, transform, fill_value=0)

    if plot:
        _plot(fig_dir, orig_curve, interpolator, msd, msd_bazin, msd_plus_bazin, msd_transformed)

    if transform:
        return interpolator, msd_transformed
    if with_bazin:
        return interpolator, msd_plus_bazin
    return interpolator, msd


def results_to_table(results, old_table, bands, table_bands, rng):
    import pandas

    df = pandas.read_csv(old_table, sep=',')
    deriv = pandas.DataFrame(index=df.index, columns=['deriv_{}'.format(band) for band in bands])
    weight_deriv = pandas.DataFrame(index=df.index, columns=['weight_deriv_{}'.format(band) for band in bands])
    logl = pandas.Series(index=df.index, name='log_likehood')
    theta = pandas.DataFrame(index=df.index, columns=['theta_{}'.format(i) for i in range(9)])
    curves = pandas.DataFrame(index=df.index,
                              columns=['{}{:+03d}'.format(band[0], int(t)) for band in table_bands for t in rng],
                              dtype=float)

    for i, (interpolator, interp_msd) in enumerate(results):
        msd = interpolator.msd
        obs_approx = interpolator(msd.arrays.x, compute_err=False)
        for j, band in enumerate(bands):
            df[band][i] = msd.odict[band].x.size
            deriv.iloc[i][j] = np.sum(np.square(msd.odict[band].y - obs_approx.odict[band].y))
            weight_deriv.iloc[i][j] = np.sum(np.square(msd.odict[band].y - obs_approx.odict[band].y)
                                             / msd.odict[band].err)
        logl[i] = interpolator.regressor.log_marginal_likelihood()
        theta.iloc[i] = interpolator.regressor.kernel_.theta
        curves.iloc[i] = np.hstack(interp_msd.odict[band].y for band in table_bands)

    df = pandas.concat((df, deriv, weight_deriv, logl, theta, curves), axis=1)
    csv_name = 'extrapol_{}_{}_{}.csv'.format(rng.min(), rng.max(), ','.join(b.replace("'", '_pr') for b in bands))
    df.to_csv(os.path.join(DATA_ROOT, csv_name), sep=',', index=False)


if __name__ == '__main__':
    import os
    import sys
    import multiprocessing
    from functools import partial
    from snad import BRI_to_gri

    band_sets = ('B,R,I', 'g,r,i', "g',r',i'",)
    if len(sys.argv) > 1:
        band_sets = sys.argv[1:]

    rng = np.r_[-20:100:121j]

    for bands in band_sets:
        bands = bands.split(',')
        basename = ','.join(band.replace("'", '_pr') for band in bands)
        sne_csv = os.path.join(DATA_ROOT, 'min3obs_{}.csv'.format(basename))
        sn_files = SNFiles(sne_csv, update=False, path=SNE_PATH)
        fig_dir = os.path.join(FIG_ROOT, basename)
        os.makedirs(fig_dir, exist_ok=True)
        transform = None
        table_bands = bands
        if set(bands) == {'B', 'R', 'I'}:
            transform = BRI_to_gri
            table_bands = ('g', 'r', 'i')
        interp = partial(interpolate,
                         rng=rng, bands=bands, peak_band=bands[1],
                         fig_dir=fig_dir, plot=True, with_bazin=False, transform=transform)
        with multiprocessing.Pool() as p:
            results = p.map(interp, sn_files.filepaths)
        results_to_table(results, sne_csv, bands, table_bands, rng)
