import sys
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name        = 'pycape',
    version     = 0.2,
    author      = 'Nick Kern',
    url         = "http://github.com/nkern/pycape",
    packages    = ['pycape']
    )

