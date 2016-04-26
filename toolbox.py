"""
toolbox.py : functions for emulating the 21cm PS and producing parameter constraints
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
from .DictEZ import create as ezcreate
import numpy.linalg as la
import astropy.io.fits as fits
import fnmatch
from plot_ellipse import plot_ellipse
import operator
from klip import klfuncs
import time
import emcee
from AttrDict import AttrDict
from .drive_21cmSense import drive_21cmSense
import scipy.optimize as opt
import corner

class workspace():

	##################################
	############ Emulator ############
	##################################

	def emu_save(self,filename=None,clobber=False):

		if filename==None:
			filename = 'emulator_%s.pkl' % '_'.join(time.asctime().split(' '))

		if os.path.isfile(filename):
			print "file exists, quitting..."
			return
		file = open(filename,'wb')
		output = pkl.Pickler(file)
		output.dump({'E':self.E})
		file.close()

	def emu_init(self,variables):
		self.E = klfuncs(variables)

	def emu_train(self,ydata_tr,param_tr,fid_ydata=None,fid_params=None,kwargs_tr={}):
		self.E.klinterp(ydata_tr,param_tr,fid_data=fid_ydata,fid_params=fid_params,**kwargs_tr)

	def emu_cross_valid(selfc,ydata_cv,param_cv,fid_ydata=None,fid_params=None):
		self.E.cross_validate(ydata_cv,param_cv,fid_data=fid_ydata,fid_params=fid_params)

	def emu_predict(self,param_pr,use_Nmodes=None):
		self.E.calc_eigenmodes(param_pr,use_Nmodes=use_Nmodes)
		return self.E.recon,self.E.recon_pos_err,self.E.recon_neg_err

	######################################
	############ Observations ############
	######################################

	def obs_init(self,dic):
		self.Obs = drive_21cmSense(dic)

	def feed_obs(self,model_kbins,obs_kbins,obs_PSdata,obs_PSerrs):
		self.Obs.x		= obs_kbins
		self.Obs.y		= obs_PSdata
		self.Obs.cov		= np.eye(self.Obs.N_data)*obs_PSerrs
		self.Obs.invcov		= la.inv(self.Obs.cov)
		self.Obs.model_kbins	= model_kbins

        #################################
        ############ Sampler ############
        #################################

	def sampler_init(self,dic,lnlike=None,lnprior=None):
		"""
		Initialize workspace self.S for sampler
		"""
		# Initialize workspace
		self.S = AttrDict(dic)

		# Check to see if an observation exists
		if 'Obs' not in self.__dict__: raise Exception("Obs class for an observation does not exist, quitting sampler...")

		# Create a model that constructs data given parameters and calculates error
		def construct_model(theta):
			# Emulate
			recon,recon_pos_err,recon_neg_err = self.emu_predict(theta,use_Nmodes=self.S.use_Nmodes)
			model		= recon
			model_err	= np.mean(np.abs([recon_pos_err,recon_neg_err]))
			# Interpolate onto observation data arrays
			self.S.model = np.interpolate(self.Obs.model_kbins,self.Obs.x,self.Obs.y)
		self.S.construct_model = construct_model

		# Specify Likelihoods, Priors and Bayes theorem numerator
		def gaussian_lnlike(theta):
			self.S.construct_model(theta)
			resid = self.Obs.y - self.S.model
			invcov = self.Obs.invcov
			return -0.5 * np.dot( resid.T, np.dot(invcov, resid) )

		def flat_lnprior(theta):
			within = True
			for i in range(self.S.N_params):
				if theta[i] < self.S.param_bounds[i][0] or theta[i] > self.S.param_bounds[i][1]:
					within = False

			if within == True:
				return np.log(1/self.S.param_hypervol)

			elif within == False:
				return -np.inf	

		# Specify loglike and logprior
		if lnlike == None:
			self.S.lnlike = gaussian_lnlike
		else:
			self.S.lnlike = lnlike	

		if lnprior == None:
			self.S.lnprior = flat_lnprior
		else:
			self.S.lnprior = lnprior

		# Specify log-probability (Bayes Theorem Numerator)
		def lnprob(theta):
			lnprior = self.S.lnprior(theta)
			lnlike = self.S.lnlike(theta)
			if not np.isfinite(lnprior):
				return -np.inf
			return lnlike + lnprior

		self.S.lnprob = lnprob

		# Initialize emcee Ensemble Sampler
		self.S.sampler = emcee.EnsembleSampler(self.S.nwalkers, self.S.ndim, self.S.lnprob)

	def find_mle(self):
		"""
		use scipy.optimize to get maximum likelihood estimate
		"""
		pass

	def drive_sampler(self,pos,step_num=500,burn_num=100):
		"""
		drive sampler
		"""
		# Run burn-in iterations
		end_pos, end_prob, end_state = self.S.sampler.run_mcmc(pos,burn_num)
		self.S.sampler.reset()	

		# Run MCMC steps
		end_pos, end_prob, end_state = self.S.sampler.run_mcmc(end_pos,step_num)



	############################################
	############ Build Training Set ############
	############################################

	def TSbuild_init(self,dic):
		self.B = drive_21cmFAST(dic)








