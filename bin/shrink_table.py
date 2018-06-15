#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas


def is_photometry_column(name):
    try:
        int(name[1:])
        return True
    except ValueError:
        return False


def is_bad_photometry_column(name, t1, t2):
    if is_photometry_column(name):
        t = int(name[1:])
        return t < t1 or t > t2
    return False


def shrink_table(table, t1, t2):
    columns = [c for c in table.columns if not is_bad_photometry_column(c, t1, t2)]
    t = table[columns]
    ph_columns = [c for c in t.columns if is_photometry_column(c)]
    good_row_idx = np.all(t[ph_columns] != 0, axis=1)
    return t.iloc[good_row_idx]


def count_against_range(table, rng):
    t1 = -rng // 3
    t2 = rng + t1
    shrinked = shrink_table(table, t1, t2)
    return shrinked.shape[0]


def shrink_file(old_file, new_file, t1, t2):
    table = pandas.read_csv(old_file)
    new_table = shrink_table(table, t1, t2)
    print(new_table.shape)
    new_table.to_csv(new_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='csv table')
    parser.add_argument('min', type=int, help='lower time range limit')
    parser.add_argument('max', type=int, help='upper time range limit')
    args = parser.parse_args()

    old_file = args.file
    t1 = args.min
    t2 = args.max
    
    new_file = os.path.join(os.path.split(old_file)[0], 'gri_pr_shrinked_{}_{}.csv'.format(t1, t2))
    shrink_file(old_file, new_file, t1, t2)


