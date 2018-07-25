#!/usr/bin/env python

import os
import shutil

import pandas


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def add_sufix(path, sufix='_uncut'):
    base, ext = os.path.splitext(path)
    new_path = base + sufix + ext
    return new_path


if __name__ == '__main__':
    for bands in ('g,r,i', 'g_pr,r_pr,i_pr'):
        table_path = os.path.join(PROJECT_ROOT, 'data/extrapol_-20.0_100.0_{}.csv'.format(bands))
        fig_dir_path = os.path.join(PROJECT_ROOT, 'fig/{}'.format(bands))

        df = pandas.read_csv(table_path, sep=',')
        ext = '.png'
        sne_from_figs = set(name[:-len(ext)] for name in os.listdir(fig_dir_path) if name.endswith(ext))
        new_df = df[df.Name.isin(sne_from_figs)]

        shutil.move(table_path, add_sufix(table_path))
        new_df.to_csv(table_path, sep=',')
