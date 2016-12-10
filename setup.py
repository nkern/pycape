import sys
import os
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name            = 'pycape',
    version         = '0.1',
    description     = 'emulator for 21cm studies',
    author          = 'Nick Kern',
    url             = "http://github.com/nkern/pycape",
    packages        = ['pycape','pycape.scripts'],
    setup_requires  = ['pytest-runner'],
    tests_require   = ['pytest']
    )


