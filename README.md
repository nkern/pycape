## pycape : Python Toolbox for Cosmic Dawn Parameter Estimation

### Version: 0.1
Code Repo : https://github.com/nkern/pycape

### About: 
**pycape** is a Python toolbox for producing 21cm parameter constraints with emulators. It contains methods for calculating a 21cm likelihood and is attached to a Markov Chain Monte Carlo sampler. The emulator code can be found in the [emupy](https://github.com/nkern/emupy) package.

### Dependencies:
pycape is dependent on a few public codes:
- numpy >= 1.10.4
- scipy >= 0.18.0
- emcee : http://dan.iel.fm/emcee/current
- emupy : https://github.com/nkern/emupy

Please check emupy for further software dependencies. 

### Installation:
To install, clone this repo and run the setup.py script as
```bash
python setup.py install
```

### Running:
See examples for demonstrations on how to run the code.

### License:
See the General Public License for details on usage

### Citation:
Please use [Kern et al. 2017](https://arxiv.org/abs/1705.04688) for citation.

### Authors:
Nicholas Kern<br>

