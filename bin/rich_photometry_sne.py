#!/usr/bin/env python

import multiprocessing
import os
from functools import partial

from thesnisright import SNFiles, OSCCurve
from thesnisright.load.snfiles import all_snad_objects
from thesnisright.load.curves import EmptyPhotometryError, NoPhotometryError


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SN_SET_ATTRS = ('claimedtype', 'redshift', 'alias')
SN_NUMBER_ATTRS = ('spectrum_count',)


def is_rich(file_path, bands):
    try:
        try:
            curve = OSCCurve.from_json(file_path, bands=bands)
        except (NoPhotometryError, EmptyPhotometryError):
            return False
        curve = curve.binned(bin_width=3, discrete_time=True)
        try:
            curve = curve.filtered()
        except EmptyPhotometryError:
            return False
        rich = curve.does_curve_have_rich_photometry(criteria={'minimum': 3})
    except Exception as e:
        print(file_path)
        raise e
    if not rich:
        return False
    attrs_dump = [curve.name]
    for attr in SN_SET_ATTRS:
        attrs_dump.append(';'.join(map(str, getattr(curve, attr))))
    for attr in SN_NUMBER_ATTRS:
        attrs_dump.append(str(getattr(curve, attr)))
    return ','.join(attrs_dump) + '\n'


def main():
    sne_path = os.path.join(PROJECT_ROOT, 'sne')
    names = all_snad_objects()
    sn_files = SNFiles(names, path=sne_path, update=False)
    for bands in ('g,r,i', "g',r',i'", 'B,R,I'):
        f = partial(is_rich, bands=bands)
        with multiprocessing.Pool() as pool:
            rich = pool.map(f, sn_files.filepaths)
        lines = (line for line in rich if line)
        data_path = os.path.join(PROJECT_ROOT, 'data')
        os.makedirs(data_path, exist_ok=True)
        csv_path = os.path.join(data_path, 'min3obs_{}.csv'.format(bands.replace("'", '_pr')))
        with open(csv_path, 'w') as f:
            f.write('Name,{}\n'.format(','.join(SN_SET_ATTRS + SN_NUMBER_ATTRS)))
            f.writelines(lines)


if __name__ == '__main__':
    main()