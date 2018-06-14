import os.path
import pickle
import re
import warnings

import numpy as np
import scipy.optimize
import pandas
import matplotlib.pyplot as plt

import curves


def construct_gri(snes):
    header = snes.keys()
    r = re.compile('^[gri]([+-]\d+)$')
    numOfDay = np.zeros(len(header))
    for i in np.arange(0, len(header)):
        m = r.match(header[i])
        numOfDay[i] = np.float(m[1]) if m else np.nan
    numOfDay = numOfDay[~np.isnan(numOfDay)]
    dayRange = np.arange(np.min(numOfDay), np.max(numOfDay) + 1)

    gri = {}
    for sne in range(len(snes)):
        rr = snes['redshifts'][sne]
        rrValue = list(map(np.float, rr.split(';'))) if rr else []
        gri[snes['SN'][sne]] = {'redshifts': rrValue}
        for band in 'gri':
            gri[snes['SN'][sne]][band] = np.vstack((dayRange, np.zeros(dayRange.shape)))

    for band in 'gri':
        for day in range(len(dayRange)):
            intDay = np.int(dayRange[day])
            column = '{}{}{:03d}'.format(band, '+' if intDay >= 0 else '-', np.abs(intDay))
            for row in range(len(snes)):
                gri[snes['SN'][row]][band][1, day] = snes[column][row]

    return gri


def filter_out_negatives(gri):
    bad_sn = set()

    for sn in gri.keys():
        for band in 'gri':
            if np.any(gri[sn][band][1, :] < 0):
                bad_sn.add(sn)

    if bad_sn:
        warnings.warn('Excluding some SNes with negative flux ({} total): {}'.format(len(bad_sn), bad_sn))

    for sn in bad_sn:
        del gri[sn]


def filter_out_redshift(gri):
    bad_sn = {sn for sn in gri if np.mean(gri[sn]['redshifts']) > 0.15}

    if bad_sn:
        warnings.warn('Excluding some SNes with large redshift: {}'.format(len(bad_sn), bad_sn))

    for sn in bad_sn:
        del gri[sn]


def crop_zeros(gri):
    for sn in gri:
        for band in 'gri':
            index = gri[sn][band][1] > 0
            gri[sn][band] = gri[sn][band][:, index]


def load_interpolated_gri_data():
    filename = 'exponential_regression.pickle'

    if os.path.isfile(filename):
        with open(filename, 'rb') as fileobject:
            return pickle.load(fileobject)

    snes = pandas.read_csv('gri.csv', keep_default_na=False)
    gri = construct_gri(snes)
    filter_out_negatives(gri)
    filter_out_redshift(gri)
    crop_zeros(gri)

    with open(filename, 'wb') as fileobject:
        pickle.dump(gri, fileobject)

    return gri


def load_gri_data():
    gri = {}

    snes = pandas.read_csv('gri.csv')
    snes = list(snes['SN'])

    try:
        files = curves.SNFiles(snes, offline=True)
    except ValueError:
        files = curves.SNFiles(snes, offline=False)

    for path in files.filepaths:
        curve = curves.OSCCurve.from_json(path)

        gri[curve.name] = {}
        for band in 'gri':
            data = curve.odict[band]
            index = ~data['isupperlimit']
            gri[curve.name][band] = np.vstack((data['x'][index], data['y'][index]))

        gri[curve.name]['redshifts'] = np.array(curve.redshift, dtype = np.float)

    filter_out_redshift(gri)

    return gri


class OneTermFitter:
    def __init__(self, p):
        self.parameters = p

    @staticmethod
    def kernel(t, t0, c0, c1, tr, tf):
        return c0 + c1 / (np.exp((t - t0) / tf) + np.exp(-(t - t0) / tr))

    @classmethod
    def fit(cls, t, x):
        c0 = np.min(x)
        c1 = 2 * (np.max(x) - c0)
        t0 = t[np.argmax(x)]
        [p, _] = scipy.optimize.curve_fit(cls.kernel, t, x, (t0, c0, c1, 5.0, 30.0))
        return cls(p)

    def __call__(self, t):
        return self.kernel(t, *self.parameters)


def plot_one_term_sne(filename, sne, fit):
    def doplot(band):
        if band in fit:
            t = sne[band][0]
            tt = np.linspace(np.min(t), np.max(t))
            plt.plot(sne[band][0], sne[band][1], '+', tt, fit[band](tt))
        else:
            plt.plot(sne[band][0], sne[band][1], '+')
    plt.figure(figsize=(6, 8))
    plt.subplot(311)
    doplot('g')
    plt.ylabel('band G')
    plt.subplot(312)
    doplot('r')
    plt.ylabel('band R')
    plt.subplot(313)
    doplot('i')
    plt.ylabel('band I')
    plt.xlabel('days')
    plt.savefig(filename)
    plt.close()


def plot_band_features(dirname, gri, features):
    for band in 'gri':
        plt.figure(figsize=(12, 9))
        # plt.axes()
        for sn in features.keys():
            if band in features[sn]:
                pars = features[sn][band].parameters
                redshift = np.mean(gri[sn]['redshifts']) if gri[sn]['redshifts'].size else 0
                plt.scatter(pars[3] * (1 + redshift), pars[4] * (1 + redshift), marker='$' + sn + '$', markersize=100)
        plt.xlabel('$\\alpha$')
        plt.ylabel('$\\beta$')
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.title('Band ' + band.upper())
        plt.savefig(os.path.join(dirname, 'band' + band + '.png'))
        plt.close()


if __name__ == '__main__':
    gri = load_gri_data()

    dirname = 'one_term'
    if not os.path.isdir(dirname):
        warnings.warn('There is no directory {}, won\'t do anything'.format(dirname))
        exit(0)

    features = {}
    for sn in gri.keys():
        features[sn] = {}
        for band in 'gri':
            try:
                data = gri[sn][band]
                if len(data[0]) < 6:
                    continue
                features[sn][band] = OneTermFitter.fit(data[0], data[1])
            except RuntimeError:
                warnings.warn('Fitting {} for {} band failed'.format(sn, band))
        plot_one_term_sne(os.path.join(dirname, sn + '.png'), gri[sn], features[sn])

    plot_band_features(dirname, gri, features)
