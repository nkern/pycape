"""
============================================================
klip : Karhunen-Loeve transformation and InterPolation
============================================================

Version 0.1

About:
- Perform Karhunen Loeve Transformation (a form of Principal Component Analysis) to compress dataset into Eigenmodes
- Produce an interpolation scheme to describe high dimensional parameter space with a relatively small training set
- Written in Python

Dependencies:
- Basic Python libraries, NumPy 1.10.4, and personal DictEZ.py code provided

Installation and Running:
- See INSTALL and RUNNING files provided

See License for details on usage

------------------------

Nicholas Kern
2016

"""

__version__	= '0.1'

from .klfuncs import *

