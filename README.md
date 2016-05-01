## pycape : PYthon toolbox for Cosmic dAwn Parameter Estimation

##### Version: 0.1
Code Repo : https://github.com/nkern/pycape

##### About: 
**pycape** is a Python toolbox for emulating the 21cm power spectrum given a training set, calculating the 21cm Likelihood given an observation (likewise forecasting performance given a mock observation), and performing Monte Carlo sampling routines within a Bayesian framework to produce realistic constraints on the parameters of Cosmic Dawn.
pycape branches the statistical and computational gap between observations of the 21cm power spectrum and the parameters that govern the physics of HII bubble growth and propagation during the Dark Ages and the Epoch of Reionization. This includes but is not limited to parameter space studies of how LCDM cosmology, high-z IGM heating mechanisms and galaxy / black hole growth affect the 21cm power spectrum over cosmic time.
pycape is specially designed to work in the regime when 1. evaluating the simulated 21cm power spectrum with a single set of parameters is time intensive (on the order of hours+), 2. the number of dimensions in one's parameter space is large (5-10+) and 3. the size of one's parameter space is vast.

##### Dependencies:
pycape is dependent on a number of public codes, including
- Standard Python 2.7 libraries
- NumPy >= 1.10.4
- SciPy >= 0.17.0
- matplotlib >= 1.5.1
- sklearn >= 0.17
- klip : https://github.com/nkern/klip
- emcee : http://dan.iel.fm/emcee/current
- 21cmSense : https://github.com/jpober/21cmSense
- aipy : https://github.com/AaronParsons/aipy

Optional:
- PyMC : http://pymc-devs.github.io/pymc
- 21cmFAST : http://homepage.sns.it/mesinger/Download.html
- OpenMPI : https://www.open-mpi.org/

##### Installation and Running:
See INSTALL for installation, and RUNNING to get started on how to use pycape

##### License:
See the General Public License for details on usage

##### Authors:
Nicholas Kern<br>

