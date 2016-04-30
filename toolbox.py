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
from sklearn.cluster import KMeans
import time
import emcee
from AttrDict import AttrDict
from .drive_21cmSense import drive_21cmSense
from .common_priors import common_priors
import scipy.optimize as opt
import corner
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

class workspace():

	def __init__(self,dic):
		""" __init__(dic) where dic is a dictionary of variables to attach to class """
		self.__dict__.update(dic)

	##################################
	############# General ############
	##################################

	def workspace_save(self,filename=None,clobber=False):
		""" workspace_save(filename=None,clobber=False) """
		if filename == None:
			filename = 'workspace_%s.pkl' % '_'.join(time.asctime().split(' '))

		if clobber == False and os.path.isfile(filename) == True:
			print 'file exists, quitting...'

		file = open(filename,'wb')
		output = pkl.Pickler(file)
		output.dump({'W':self.__dict__})
		file.close()

	def print_message(self,string,type=1):
		""" print_message(string,type=1) """
		if type == 1:
			print ''
			print string
			print '-'*40

	##################################
	############ Emulator ############
	##################################

	def emu_save(self,filename=None,clobber=False):
		"""emu_save(filename=None,clobber=False)"""

		if filename==None:
			filename = 'emulator.pkl'

		if os.path.isfile(filename) == True and clobber == False:
			print "file exists, quitting..."
			return

		file = open(filename,'wb')
		output = pkl.Pickler(file)
		output.dump({'E':self.E})
		file.close()

		if filename==None:
			filename='emulator.pkl'
		file = open(filename,'rb')
		input = pkl.Unpickler(file)
		dic = input.load()
		self.E.__dict__.update(dic)
		file.close()

	def emu_load(self,filename=None):
		"""emu_load(filename=None)"""
		if filename == None:
			filename = 'emulator.pkl'
		file = open(filename,'rb')
		input = pkl.Unpickler(file)
		self.E = input.load()
		file.close()

	def emu_init(self,variables):
		"""emu_init(variables) where variables is a dict w/ vars to attach to emulator class E"""
		self.E = klfuncs(variables)

	def emu_cluster(self,grid,R_mult=1.2,tree_kwargs={},kmeans_kwargs={}):
		"""break the training set into 2**ndim sub-space via KMeans"""
		# Transform into Cholesky Basis
		cov = np.inner(grid.T,grid.T)/grid.shape[0]
		L = la.cholesky(cov)
		invL = la.inv(L)
		grid = np.dot(invL,grid)

		# First construct Tree
		self.E.create_tree(grid,**tree_kwargs)

		# Do clustering
		self.E.kmeans = KMeans(**kmeans_kwargs)
		self.E.kmeans.fit(grid)

		# Get distance each cluster center is from origin
		self.E.cluster_cent = self.E.kmeans.cluster_centers_
		self.E.kmeans.cluster_R_ = np.array(map(la.norm,self.E.cluster_cent))

		# Give each cluster an ID
		cluster_num = len(self.E.cluster_cent)
		self.E.cluster_ID = np.arange(cluster_num)

		# Assign each cell a training set based on points within a distance of cluster_R * R_mult
		self.E.cluster_TS = []
		for i in range(cluster_num):
			within_r = self.E.kmeans.query_radius(grid, r = self.E.cluster_cent[i] * R_mult)
			self.E.cluster_TS.append(np.dot(L,grid[within_r]))

		self.E.cluster_TS = np.array(self.E.cluster_TS)

		# Transform cluster centers into original space
		self.E.cluster_cent = np.dot(L,self.E.cluster_cent)
		self.E.L, self.E.invL = L, invL

	def emu_get_closest_cluster(self,X,k=1):
		""" get k closest clusters """

		# Get Euclidean distance
		cluster_dist = np.array(map(la.norm,self.E.cluster_cent-X))
		
		# Sort by distance
		close_IDs = self.E.cluster_ID[np.argsort(cluster_dist)][::-1]

		return close_IDs[:k]

	def emu_train(self,data_tr,param_tr,fid_data=None,fid_params=None,kwargs_tr={}):
		"""
		emu_trin(data_tr,param_tr,fid_data=None,fid_params=None,kwargs_tr={})
		data_tr		: [N_samples,N_data]
		param_tr	: [N_samples,N_params]
		fid_data	: [N_samples,]
		fid_params	: [N_params,]
		kwargs_tr	: kwargs to pass to klinterp() 
		"""
		self.E.data_tr = data_tr
		self.E.param_tr = param_tr
		self.E.fid_data = fid_data
		self.E.fid_params = fid_params
		self.E.klinterp(data_tr,param_tr,fid_data=fid_data,fid_params=fid_params,**kwargs_tr)

	def emu_cross_valid(selfc,data_cv,param_cv,fid_data=None,fid_params=None):
		self.E.cross_validate(data_cv,param_cv,fid_data=fid_data,fid_params=fid_params)

	def emu_predict(self,param_pr,use_Nmodes=None):
		self.E.calc_eigenmodes(param_pr,use_Nmodes=use_Nmodes)
		return self.E.recon,self.E.recon_pos_err,self.E.recon_neg_err

	def emu_forwardprop_weighterr(self,theta,use_Nmodes=None):
		if use_Nmodes == None: use_Nmodes = self.E.N_modes
		recon,recon_pos_err,recon_neg_err = self.emu_predict(theta,use_Nmodes=use_Nmodes)
		model		= recon.T[self.E.model_lim].T
		model_err	= np.abs(np.array(map(lambda x: map(np.mean,x),map(lambda x: np.array(x).T,zip(recon_pos_err.T[self.E.model_lim].T,recon_pos_err.T[self.E.model_lim].T)))))
		return model, model_err

	######################################
	############ Observations ############
	######################################

	def obs_init(self,dic):
		self.Obs = drive_21cmSense(dic)

	def obs_feed(self,model_kbins,obs_kbins,obs_PSdata,obs_PSerrs):
		self.Obs.x		= obs_kbins
		self.Obs.y		= obs_PSdata
		self.Obs.y_err		= obs_PSerrs
		self.Obs.cov		= np.eye(self.Obs.N_data)*self.Obs.y_err
		self.Obs.invcov		= la.inv(self.Obs.cov)
		self.Obs.model_kbins	= model_kbins

	def obs_update_cov(self,cov_add):
		self.Obs.cov += cov_add
		self.Obs.invcov = la.inv(self.Obs.cov)

	def obs_save(self,filename,clobber=False):
                if filename == None:
                        filename = 'observation_%s.pkl' % '_'.join(time.asctime().split(' '))

                if clobber == False and os.path.isfile(filename) == True:
                        print 'file exists, quitting...'

                file = open(filename,'wb')
                output = pkl.Pickler(file)
                output.dump({'Obs':self.Obs})
                file.close()
	

        #################################
        ############ Sampler ############
        #################################

	def samp_construct_model(self,theta,add_model_err=False):
		# Emulate
		recon,recon_pos_err,recon_neg_err = self.emu_predict(theta,use_Nmodes=self.S.use_Nmodes)
		model_init                   = recon[0][self.E.model_lim]
		model_err_init               = np.array(map(np.mean, np.abs([recon_pos_err[0][self.E.model_lim],recon_neg_err[0][self.E.model_lim]]).T))

		# Interpolate model onto observation data arrays
		model = []
		model_err = []
		for i in range(len(self.Obs.x)):
			model.extend( np.interp(self.Obs.x[i],self.Obs.model_kbins,model_init[i]) )
			model_err.extend( np.interp(self.Obs.x[i],self.Obs.model_kbins,model_err_init[i]) )

		self.S.model            = np.array(model)
		self.S.model_err        = np.array(model_err)

		# If add model error is true, add diagonal of covariance and model errs in quadrature
		if add_model_err == True:
			self.S.data_cov		= self.Obs.cov + np.eye(self.Obs.N_data)*self.S.model_err
			self.S.data_invcov	= la.inv(self.S.data_cov)
		else:
			self.S.data_cov		= self.Obs.cov
			self.S.data_invcov	= self.Obs.invcov
	
	def samp_gaussian_lnlike(self,theta,add_model_err=False):
		self.samp_construct_model(theta,add_model_err=add_model_err)
		resid = self.Obs.y - self.S.model
		return -0.5 * np.dot( resid.T, np.dot(self.S.data_invcov, resid) )

	def samp_flat_lnprior(self,theta):
		within = True
		for i in range(self.S.N_params):
			if theta[i] < self.S.param_bounds[i][0] or theta[i] > self.S.param_bounds[i][1]:
				within = False
		if within == True:
			return np.log(1/self.S.param_hypervol)
		elif within == False:
			return -np.inf

	def samp_lnprob(self,theta,add_model_err=False):
		lnprior = self.S.lnprior(theta)
		lnlike = self.S.lnlike(theta, add_model_err=add_model_err)
		if not np.isfinite(lnprior):
			return -np.inf
		return lnlike + lnprior


	def samp_init(self, dic, lnlike=None, lnprior=None, lnprob_kwargs={}, sampler_kwargs={}):
		"""
		Initialize workspace self.S for sampler
		"""
		# Initialize workspace
		self.S = AttrDict(dic)

		# Check to see if an observation exists
		if 'Obs' not in self.__dict__: raise Exception("Obs class for an observation does not exist, quitting sampler...")

		# Specify loglike and logprior, can feed your own, but if not use default
		if lnlike == None:
			self.S.lnlike = self.samp_gaussian_lnlike
		else:
			self.S.lnlike = lnlike	

		if lnprior == None:
			self.S.lnprior = self.samp_flat_lnprior
		else:
			self.S.lnprior = lnprior

		# Specify log-probability (Bayes Theorem Numerator)
		self.S.lnprob = self.samp_lnprob

		# Initialize emcee Ensemble Sampler
		self.S.sampler = emcee.EnsembleSampler(self.S.nwalkers, self.S.ndim, self.S.lnprob, kwargs=lnprob_kwargs, **sampler_kwargs)

	def samp_mle(self):
		"""
		use scipy.optimize to get maximum likelihood estimate
		"""
		pass

	def samp_drive(self,pos,step_num=500,burn_num=100):
		"""
		drive sampler
		"""
		if burn_num > 0:
			# Run burn-in iterations
			end_pos, end_prob, end_state = self.S.sampler.run_mcmc(pos,burn_num)
			self.S.sampler.reset()	

		# Run MCMC steps
		if burn_num > 0:
			end_pos, end_prob, end_state = self.S.sampler.run_mcmc(end_pos,step_num)
		else:
			end_pos, end_prob, end_state = self.S.sampler.run_mcmc(pos,step_num)


	def samp_drive_mpi(self,pos,step_num=500,burn_num=100,mpi_np=5,sampler_init_kwargs={},lnprob_kwargs={},sampler_kwargs={},workspace=None):
		"""
		drive sampler using mpirun
		"""
		if workspace == None:
			raise Exception("Didn't feed a workspace")

		# Save workspace
		self.sampler_init_kwargs = sampler_init_kwargs
		self.sampler_kwargs =  sampler_kwargs
		self.lnprob_kwargs = lnprob_kwargs
		self.burn_num = burn_num
		self.step_num = step_num
		self.pos = pos
		file = open('Workspace.pkl','wb')
		output = pkl.Pickler(file)
		output.dump({'W':workspace})
		file.close()

		# Use mpirun to run in parallel
		os.system('mpirun -np %s python %s/drive_sampler_mpi.py' % (mpi_np,self.dir_pycape))

		# Initialize Sampler
		self.sampler_init(sampler_init_kwargs)

		# Load in chains
		for i in range(mpi_np):
			file = open(self.dir_pycape+'/mpi_chains/mpi_chain_rank%s.pkl'%i,'rb')
			input = pkl.Unpickler(file)
			self.S.sampler.__dict__.update(input.load())
			file.close()
			if i == 0:
				self.S.sampler.mpi_chain = self.S.sampler.rank0_chain
			else:
				self.S.sampler.mpi_chain = np.vstack([self.S.sampler.mpi_chain,self.S.sampler.__dict__['rank%s_chain'%i]])

        def samp_save(self,filename,clobber=False):
                if filename == None:
                        filename = 'sampler_%s.pkl' % '_'.join(time.asctime().split(' '))
        
                if clobber == False and os.path.isfile(filename) == True:
                        print 'file exists, quitting...'

                file = open(filename,'wb')
                output = pkl.Pickler(file)
                output.dump({'S':self.S})
                file.close()


	def samp_predict_newTS(self):
		pass

	def samp_common_priors_init(self,dic):
		self.CP = common_priors(dic)

	############################################
	############ Build Training Set ############
	############################################

	def TSbuild_init(self,dic):
		self.B = drive_21cmFAST(dic)


        def TSbuilder_save(self,filename,clobber=False):
                if filename == None:
                        filename = 'TSbuilder_%s.pkl' % '_'.join(time.asctime().split(' '))
        
                if clobber == False and os.path.isfile(filename) == True:
                        print 'file exists, quitting...'

                file = open(filename,'wb')
                output = pkl.Pickler(file)
                output.dump({'B':self.B})
                file.close()



	############################################
	################# Plotting #################
	############################################

	def corner_plot(self,levels=None):
		if levels == None:
			levels = [0.34,0.68,0.90,0.95]
		fig = corner.corner(samples.T[::-1].T, labels=p_latex[::-1], truths=p_true[::-1], range=param_bounds[::-1])




