## pycape : Python toolbox for Cosmic Dawn Parameter Estimation
[![Build Status](https://travis-ci.com/nkern/pycape.svg?token=5USCxbBe7R1gkSvyQwzK&branch=master)](https://travis-ci.com/nkern/pycape)

### Version: 0.1
Code Repo : https://github.com/nkern/pycape

### About: 
**pycape** is a Python toolbox for emulating the 21cm power spectrum given a training set, calculating the 21cm Likelihood given an observation (likewise forecasting performance given a mock observation), and performing Monte Carlo sampling routines within a Bayesian framework to produce realistic 21cm constraints on the parameters of Cosmic Dawn.
pycape is specially designed to work in the regime when 1. evaluating the simulated 21cm power spectrum with a single set of parameters is time intensive (on the order of hours+), 2. the number of dimensions in one's parameter space is large (5-10+) and 3. the size of one's parameter space is vast.

### Dependencies:
pycape is dependent on a number of public codes.
- NumPy >= 1.10.4
- SciPy >= 0.18.0
- matplotlib >= 1.5.1
- sklearn >= 0.18
- emcee : http://dan.iel.fm/emcee/current
- astropy >= 1.2.1
- aipy : https://github.com/AaronParsons/aipy

The packages above can be pip installed.

- 21cmSense : https://github.com/nkern/21cmSense

You can install 21cmSense by cloning the above repo and then adding the 21cmSense directory to your PYTHONPATH
by adding the following to .bashrc
```bash
export PYTHONPATH=<where_21cmSense_lives>/21cmSense:$PYTHONPATH
```
### Download
To download, simply clone this repo
```bash
git clone https://github.com/nkern/pycape
```

### Installation:
To install, cd into the downloaded pycape repo and run the setupy.py script as
```bash
python setup.py install
```
### Running:
pycape is not an end-to-end analysis package; It is a collection of tools one can combine to meet their science goal.

### License:
See the General Public License for details on usage

### Authors:
Nicholas Kern<br>

