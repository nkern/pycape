"""
## pycape : Python Toolbox for Cosmic Dawn Parameter Estimation

##### Version: 0.1
Code Repo : https://github.com/nkern/pycape


##### About: 
**pycape** is a Python toolbox for emulating the 21cm power spectrum given a training set, calculating the 21cm Likelihood given an observation (likewise forecasting performance given a mock observation), and performing Monte Carlo sampling routines within a Bayesian framework to produce realistic constraints on the parameters of Cosmic Dawn.
pycape is specially designed to work in the regime when 1. evaluating the simulated 21cm power spectrum with a single set of parameters is time intensive (on the order of hours+), 2. the number of dimensions in one's parameter space is large (5-10+) and 3. the size of one's parameter space is vast.

##### Dependencies:
pycape is dependent on a number of public codes, including
- Standard Python 2.7 libraries
- NumPy >= 1.10.4
- SciPy >= 0.17.0
- matplotlib >= 1.5.1
- sklearn >= 0.17
- klip : https://github.com/nkern/klip
- pystan : https://pystan.readthedocs.org/en/latest/

Optional:
- emcee : http://pymc-devs.github.io/pymc/
- PyMC : http://dan.iel.fm/emcee/current/
- 21cmFAST : 
- 21cmSense : 

##### Installation and Running:


##### License:
See the GPL License for details on usage

##### Authors:
Nicholas Kern
"""
from .emulator import *
from .sampler import *
from .observations import *
from .simulations import *
import os

__path__ = os.path.dirname(os.path.realpath(__file__))

