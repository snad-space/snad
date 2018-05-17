#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from curves import SNCurve

JSON_PATH = 'test_binning.json'
BAND = 'X'
PNG_PATH = 'test_binning.png'


def plot(lc, size=1):
     ul = lc[lc['isupperlimit']]
     d = lc[~lc['isupperlimit'] & np.isnan(lc['e_flux'])]

     ed = lc[~lc['isupperlimit'] & np.isfinite(lc['e_flux'])]
     plt.yscale('log')
     plt.plot(ul['time'], ul['flux'], 'v', ms=size)
     plt.plot(d['time'], d['flux'], 'x', ms=size)
     plt.errorbar(ed['time'], ed['flux'], ed['e_flux'], fmt='+', ms=size)


if __name__ == '__main__':
    plot(SNCurve.from_json(JSON_PATH)[BAND], 6)
    plot(SNCurve.from_json(JSON_PATH, bin_width=1)[BAND], 12)
    plt.xticks(range(9))
    plt.grid()
    plt.savefig(PNG_PATH)
