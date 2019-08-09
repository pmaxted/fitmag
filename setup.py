#!/usr/bin/env python
# Adapted from https://github.com/pypa/sampleproject/blob/master/setup.py
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# VERSION
try:
    from version import __version__
except:
    __version__ = ''

setup(
    name='fitmag',
    version=__version__,
    author='Pierre Maxted',
    author_email='p.maxted@keele.ac.uk',
    license='GNU GPLv3',
    description='Analysis of SDSS/2MASS/WISE photometry for stars',
    long_description=long_description,
    classifiers = [
        'Development Status :: 1 - Planning Development Status ',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python'
    ],
    packages=['fitmag'],
    package_data={ 'fitmag': ['data/g_Teff_zams_tams.dat'] }, 
    entry_points={
      'console_scripts': [
        'fitmag=fitmag.fitmag:main',
      ],
    },
)

