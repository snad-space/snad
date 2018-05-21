#!/usr/bin/env python

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from curves import OSCCurve

JSON_PATH = 'test_binning.json'
BAND = 'X'
PNG_PATH = 'test_binning.png'


def plot(lc, size=1):
     ul = lc[lc['isupperlimit']]
     d = lc[~lc['isupperlimit'] & np.isnan(lc['err'])]

     ed = lc[~lc['isupperlimit'] & np.isfinite(lc['err'])]
     plt.yscale('log')
     plt.plot(ul['x'], ul['y'], 'v', ms=size)
     plt.plot(d['x'], d['y'], 'x', ms=size)
     plt.errorbar(ed['x'], ed['y'], ed['err'], fmt='+', ms=size)


if __name__ == '__main__':
    curve = OSCCurve.from_json(JSON_PATH)
    plot(curve.binned(bin_width=1)[BAND], 12)
    plot(curve[BAND], 6)
    plt.xticks(range(9))
    plt.grid()
    plt.savefig(PNG_PATH)
