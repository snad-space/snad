#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='snad',
    license='MIT',
    author='Konstantin Malanchev',
    author_email='malanchev@physics.msu.ru',
    packages=find_packages(exclude=['test']),
    test_suite='test',
)
