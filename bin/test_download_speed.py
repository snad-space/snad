#!/usr/bin/env python

import time
import shutil

from thesnisright import SNFiles


small_files = [

]


def download(baseurl, sns):
    t = time.monotonic()
    SNFiles(sns, baseurl=baseurl)
    dur = time.monotonic() - t
    print('{} downloaded from {} in {:.3f} seconds'.format(sns, baseurl, dur))


def download_from(baseurl):
    try:
        shutil.rmtree('sne')
    except FileNotFoundError:
        pass

    for sns in ['small.txt', ['SN1987A']]:
        download(baseurl, sns)


if __name__ == '__main__':
    for baseurl in ['https://sne.space/sne/', 'https://snad.sai.msu.ru/sne/', 'http://snad.sai.msu.ru/sne/']:
        download_from(baseurl)
