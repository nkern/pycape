"""
pycape.py : functions for emulating the 21cm PS and producing parameter constraints

"""
import os
import numpy as np
import cPickle as pkl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from fits_table import fits_table, fits_data, fits_append
import scipy.stats as stats
from grid_helper import *
from DictEZ import create as ezcreate
import numpy.linalg as la
import astropy.io.fits as fits
import fnmatch
from plot_ellipse import plot_ellipse
import operator
from klip import klfuncs
import time

class pycape(object):

	########################################
	############ Emulator Specs ############
	########################################

	def __init__(self,variables):
		self.E = klfuncs(variables)

	def save_emu(self,filename=None,clobber=False):

		if filename==None:
			filename = 'emulator_%s.pkl' % '_'.join(time.asctime().split(' '))

		if os.path.isfile(filename):
			print "file exists, quitting..."
			return
		file = open(filename,'wb')
		output = pkl.Pickler(file)
		output.dump({'E':self.E})
		file.close()


	##################################################
	############ Emulator Train & Predict ############
	##################################################

	def init(self,variables):
		self.E = self.klfuncs(variables)


	def train(self,ydata_tr,param_tr,fid_ydata=None,fid_params=None,kwargs_tr={}):
		self.E.klinterp(ydata_tr,param_tr,fid_data=fid_ydata,fid_params=fid_params,**kwargs_tr)


	def predict(self,param_pr):
		self.E.calc_eigenmodes



	def gauss_like(self,data,model,covar,ln=True):
		resid = data - model
		if log == True:
			return -0.5 * np.dot( resid.T, np.dot(la.inv(covar), resid) )
		else:
			return np.exp( -0.5 * np.dot( resid.T, np.dot(la.inv(covar), resid) ) )
















