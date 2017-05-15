## pycape : Python Toolbox for Cosmic Dawn Parameter Estimation
[![Build Status](https://travis-ci.com/nkern/pycape.svg?token=5USCxbBe7R1gkSvyQwzK&branch=master)](https://travis-ci.com/nkern/pycape)

### Version: 0.1
Code Repo : https://github.com/nkern/pycape

### About: 
**pycape** is a Python toolbox for emulating the 21cm power spectrum given a training set, calculating the 21cm likelihood given an observation (likewise forecasting performance given a mock observation), and performing Monte Carlo sampling routines within a Bayesian framework to produce realistic 21cm constraints on the parameters of Cosmic Dawn.
pycape is specially designed to work in the regime when evaluating the simulated 21cm power spectrum with a single set of parameters is time intensive (on the order of minutes - hours+). 

### Dependencies:
pycape is dependent on a number of public codes, which can be pip installed
- numpy >= 1.10.4
- scipy >= 0.18.0
- emcee : http://dan.iel.fm/emcee/current

### Installation:
To install, clone this repo, cd into it and run the setupy.py script as
```bash
python setup.py install
```
### Running:
pycape is not an end-to-end analysis package

### License:
See the General Public License for details on usage

### Authors:
Nicholas Kern<br>

