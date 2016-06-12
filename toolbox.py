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
from sklearn.gaussian_process import GaussianProcess
from sklearn.cluster import KMeans
from sklearn import neighbors
import time
import emcee
from AttrDict import AttrDict
from .drive_21cmSense import drive_21cmSense
from .drive_camb import drive_camb
from .common_priors import *
import scipy.optimize as opt
import corner
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

class workspace(object):

	def __init__(self,dic):
		""" __init__(dic) where dic is a dictionary of variables to attach to class """
		self.__dict__.update(dic)

	##################################
	############# General ############
	##################################

	def workspace_save(self,filename=None,clobber=False):
		""" workspace_save(filename=None,clobber=False) """
		if filename is None:
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
		if filename is None:
			filename = 'emulator.pkl'
		file = open(filename,'rb')
		input = pkl.Unpickler(file)
		self.E = input.load()
		file.close()

	def emu_init(self,variables,emu_name=None):
		"""emu_init(variables) where variables is a dict w/ vars to attach to emulator class E"""
		if emu_name is None:
			self.E = klfuncs(variables)
		else:
			self.__dict__[emu_name] = klfuncs(variables)

	def emu_cluster(self,grid,R_mult=1.2,tree_kwargs={},kmeans_kwargs={}):
		"""break the training set into clusters via KMeans"""
		# Transform into Cholesky Basis
		cov = np.inner(grid.T,grid.T)/grid.shape[0]
		L = la.cholesky(cov)
		invL = la.inv(L)
		grid = np.dot(invL,grid.T).T

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
		self.E.clus_TS = []
		for i in range(cluster_num):
			within_r = self.E.tree.query_radius(self.E.cluster_cent[i], r = self.E.kmeans.cluster_R_[i] * R_mult)[0]
			self.E.clus_TS.append(within_r)

		self.E.clus_TS = np.array(self.E.clus_TS)

		# Transform cluster centers into original space
		self.E.kmeans.L, self.E.kmeans.invL = L, invL

	def emu_get_closest_clusters(self,X,k=1):
		""" get k closest clusters """
		# Transform to cholesky space
		X = np.dot(self.E.kmeans.invL,X)

		# Get Euclidean distance
		cluster_dist = np.array(map(la.norm,self.E.cluster_cent-X))
		
		# Sort by distance
		sort = np.argsort(cluster_dist)
		cluster_dist = cluster_dist[sort]
		close_IDs = self.E.cluster_ID[sort]

		return close_IDs[:k],cluster_dist[:k]

	def emu_train(self,data_tr,param_tr,fid_data=None,fid_params=None,kwargs_tr={},emu=None):
		"""
		emu_trin(data_tr,param_tr,fid_data=None,fid_params=None,kwargs_tr={},emu=None)

		data_tr		: [N_samples,N_data]
		param_tr	: [N_samples,N_params]
		fid_data	: [N_samples,]
		fid_params	: [N_params,]
		kwargs_tr	: kwargs to pass to interp() 
		"""
		if emu is None:
			self.E.interp(data_tr,param_tr,fid_data=fid_data,fid_params=fid_params,**kwargs_tr)
		else:
			emu.interp(data_tr,param_tr,fid_data=fid_data,fid_params=fid_params,**kwargs_tr)

	def emu_predict(self,param_pr,**kwargs):
		self.E.predict(param_pr,**kwargs)

	def emu_forwardprop_weighterr(self,theta,use_Nmodes=None):
		if use_Nmodes is None: use_Nmodes = self.E.N_modes
		recon,recon_pos_err,recon_neg_err = self.emu_predict(theta,use_Nmodes=use_Nmodes)
		model		= recon.T[self.E.model_lim].T
		model_err	= np.abs(np.array(map(lambda x: map(np.mean,x),map(lambda x: np.array(x).T,zip(recon_pos_err.T[self.E.model_lim].T,recon_pos_err.T[self.E.model_lim].T)))))
		return model, model_err

	######################################
	############ Observations ############
	######################################

	def obs_init(self,dic):
		self.Obs = drive_21cmSense(dic)

	def obs_feed(self,model_xbins,obs_xbins,obs_ydata,obs_yerrs):
		self.Obs.x		= obs_xbins	# mock obs x data (kbins)
		self.Obs.y		= obs_ydata	# mock obs y data (deldel)
		self.Obs.y_errs		= obs_yerrs	# mock obs y errs (sensitivity)
		self.Obs.cov		= np.eye(self.Obs.N_data)*self.Obs.y_errs**2
		self.Obs.invcov		= la.inv(self.Obs.cov)
		self.Obs.model_xbins	= model_xbins	# simulation x data (kbins)
		self.Obs.model_shape	= model_xbins.shape

	def obs_update_cov(self,cov_add):
		self.Obs.cov += cov_add
		self.Obs.invcov = la.inv(self.Obs.cov)

	def obs_save(self,filename,clobber=False):
                if filename is None:
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

	def samp_construct_model(self,theta,add_model_err=False,calc_lnlike_emu_err=False,fast=False,LAYG=True,LAYG_pretrain=False,
					emu_err_mc=False,GPhyperNN=False,k=50,kwargs_tr={},predict_kwargs={},**kwargs):
		# LAYG
		if LAYG == True:
			parsph = np.dot(self.E.invL,np.array([theta-self.E.fid_params]).T).T[0]
			grid_D, grid_NN = self.E.tree.query(parsph,k=k)
			grid_NN = grid_NN[0]
			grid_D = grid_D[0]
			if GPhyperNN == True:
				weight_func = lambda x: 1/x
				weights = weight_func(grid_D)
				weights /= sum(weights)
				theta0 = sum(self.E.GPhyperParams[grid_NN]*weights)
				kwargs_tr['theta0'] = theta0
				kwargs_tr['thetaL'] = None
				kwargs_tr['thetaU'] = None

			if LAYG_pretrain == True:
				kwargs_tr['theta0'] = self

			self.emu_train(self.E.data_tr[grid_NN],self.E.grid_tr[grid_NN],fid_data=self.E.fid_data,fid_params=self.E.fid_params,kwargs_tr=kwargs_tr)

		# Emulate
		self.emu_predict(theta,**predict_kwargs)
		recon		= self.E.recon[0]
		recon_err	= self.E.recon_err
		self.samp_interp_mod2obs(recon,recon_err,add_model_err=add_model_err,calc_lnlike_emu_err=calc_lnlike_emu_err,emu_err_mc=emu_err_mc)

	def samp_interp_mod2obs(self,recon,recon_err,add_model_err=False,calc_lnlike_emu_err=False,cut_high_fracerr=100.0,emu_err_mc=False,**kwargs):
		""" samp_interp_mod2obs(recon,recon_err,add_model_err=True)
		- interpolate model data in simulation basis to observation basis (x-axis points)
		"""
		model_predic = recon[self.E.model_lim].reshape(self.Obs.model_shape)
		model_err_predic = recon_err[self.E.model_lim].reshape(self.Obs.model_shape)

		# Interpolate model onto observation data arrays
		model = []
		model_err = []
		for i in range(self.S.z_num):
			model.extend( np.interp(self.Obs.x[i],self.Obs.model_xbins[i],model_predic[i]) )
			model_err.extend( np.interp(self.Obs.x[i],self.Obs.model_xbins[i],model_err_predic[i]) )

		self.S.model            = np.array(model).ravel()
		self.S.model_err        = np.array(model_err).ravel()

		# Resample model from Gaussian with scale of model_err if emu_err_mc = True
		if emu_err_mc == True:
			resampled_model = np.array([stats.norm.rvs(loc=self.S.model[i],scale=self.S.model_err[i],size=1)[0] for i in range(len(self.S.model))])
			self.S.model = resampled_model

		# If add model error is true, add diagonal of covariance and model errs in quadrature
		if add_model_err == True:
			self.S.data_cov		= self.Obs.cov + np.eye(self.Obs.cov.shape[0])*self.S.model_err**2
			self.S.data_invcov	= la.inv(self.S.data_cov)
		else:
			self.S.data_cov		= np.copy(self.Obs.cov)
			self.S.data_invcov	= np.copy(self.Obs.invcov)

		# Calculate uncertainty in lnlikelihood estimate purely from emulator error
		if calc_lnlike_emu_err == True:
			resid = self.Obs.y - self.S.model
			self.S.lnlike_emu_err_i = 0.5*np.sqrt(2) * resid**2 / self.S.data_cov.diagonal() * self.S.model_err / resid
			self.S.lnlike_emu_err = np.sqrt( sum(self.S.lnlike_emu_err_i[self.S.model_err/self.S.model<cut_high_fracerr]**2) )
	
	def samp_gaussian_lnlike(self,ydata,model,invcov):
		resid = ydata - model
		return -0.5 * np.dot( resid.T, np.dot(invcov, resid) )

	def samp_flat_lnprior(self,param_bounds):
		"""
		- Initialize a flat prior function
		"""
		def flat_lnprior(theta,param_bounds=param_bounds):
			within = True
			if theta < param_bounds[0] or theta > param_bounds[1]:
				within = False
			if within == True:
				return np.log(1/(param_bounds[1]-param_bounds[0]))
			else:
				return -np.inf

		self.S.lnprior_funcs.append(flat_lnprior)

	def samp_gauss_lnprior(self,mean,sigma,return_func=False):
		"""
		- Initialize a Gaussian prior function
		"""
		def gauss_lnprior(theta,mean=mean,sigma=sigma):
			return np.log(stats.norm.pdf(theta,loc=mean,scale=sigma))
	
		if return_func == True:
			return gauss_lnprior
		else:	
			self.S.lnprior_funcs.append(gauss_lnprior)

	def samp_lnprior(self,theta):
		"""
		- Call the previously created self.S.lnpriors list that holds the prior functions for each parameter
		"""
		lnprior = 0
		for i in range(len(theta)):
			lnprior += self.S.lnprior_funcs[i](theta[i])
		return lnprior

	def samp_lnprob(self,theta,**lnlike_kwargs):
		self.samp_construct_model(theta,**lnlike_kwargs)
		lnlike = self.S.lnlike(self.Obs.y,self.S.model,self.S.data_invcov)
		lnprior = self.S.lnprior(theta)
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
		if lnlike is None:
			self.S.lnlike = self.samp_gaussian_lnlike
		else:
			self.S.lnlike = lnlike	

		if lnprior is None:
			self.S.lnprior_funcs = []
			# Initialize flat priors for all parameters
			for i in range(self.S.N_params):
				self.samp_flat_lnprior(self.S.param_bounds[i])
		else:
			self.S.lnprior_funcs = lnprior

		self.S.lnprior = self.samp_lnprior

		# Specify log-probability (Bayes Theorem Numerator)
		self.S.lnprob = self.samp_lnprob

		# Initialize emcee Ensemble Sampler
		self.S.sampler = emcee.EnsembleSampler(self.S.nwalkers, self.S.ndim, self.S.lnprob, kwargs=lnprob_kwargs, **sampler_kwargs)

	def samp_mle(self):
		"""
		use scipy.optimize to get maximum likelihood estimate
		"""
		pass

	def samp_cross_valid(self,grid_cv,data_cv,lnlike_kwargs={},also_record=[]):
		"""
		samp_cross_valid()
		- get error on emulated power spectra or likelihood
		"""
		grid_len = len(grid_cv)
		emu_lnlike = []
		tru_lnlike = []
		other_vars = dict(zip(also_record,[[] for ii in range(len(also_record))]))
		for i in range(grid_len):
			self.samp_construct_model(grid_cv[i],**lnlike_kwargs)
			e_lnl = self.S.lnlike(self.Obs.y,self.S.model,self.S.data_invcov)
			emu_lnlike.append(e_lnl)

			for name in also_record:
				other_vars[name].append(self.S.__dict__[name])

			self.samp_interp_mod2obs(data_cv[i],np.zeros(len(data_cv[i])),**lnlike_kwargs)
			t_lnl = self.S.lnlike(self.Obs.y,self.S.model,self.S.data_invcov)
			tru_lnlike.append(t_lnl)

		emu_lnlike = np.array(emu_lnlike)
		tru_lnlike = np.array(tru_lnlike)
		return emu_lnlike, tru_lnlike, other_vars

	def samp_marginalized_pdf(self,grid_cv,data_cv,samples,theta0=1.0,nugget=1e-4,
					lnprob_kwargs={},marg_ax=0,hist_bins=1000):
		"""
		solve for marginalized pdf given samples of the joint_pdf and a cross validation set
		"""
		# Find dimensions
		ndim = len(grid_cv.T)

                # Get MAP
		def get_map(samples,grid_bounds):
                	hist_data = np.histogram(samples,range=(grid_bounds[0],grid_bounds[1]),bins=25)
                	dx = hist_data[1][1] - hist_data[1][0]
                	xpoints = hist_data[1][1:] - dx/2.
			pdf = hist_data[0]

			# Get robust measurement of center of pdf
			x_range = xpoints
			three_quarter_max = np.max(pdf)*3./4.
			xmax = x_range[np.where(pdf==np.max(pdf))]
			xrange_plus_sel = np.where(x_range>xmax)[0]
			xrange_neg_sel = np.where(x_range<xmax)[0]
			x_tq_plus = x_range[xrange_plus_sel][np.argsort(np.abs(pdf[xrange_plus_sel]-three_quarter_max))][0]
			x_tq_neg = x_range[xrange_neg_sel][np.argsort(np.abs(pdf[xrange_neg_sel]-three_quarter_max))][0]
			pdf_cent = np.mean([x_tq_plus,x_tq_neg])
			return pdf_cent

		pdf_cent = np.zeros(ndim)
		for i in range(ndim): pdf_cent[i] = get_map(samples.T[i],grid_bounds[i])

		# Whiten cross validation set
		Xsph = np.dot(self.E.invL,(grid_cv-self.E.fid_params).T).T

		# Get exact error over cross validation set
		emu_lnlike,true_lnlike,o_vars = self.samp_cross_valid(grid_cv,data_cv,lnlike_kwargs=lnprob_kwargs)
		lnlike_err = np.abs(emu_lnlike-true_lnlike)

		# GP Regression for the lnlike_err
		gp_kwargs = {'regr':'linear','theta0':theta0,'thetaL':None,'thetaU':None,
                'random_start':1,'verbose':False,'corr':'squared_exponential','nugget':nugget}

		GP = GaussianProcess(**gp_kwargs).fit(Xsph,lnlike_err)

		## Predict lnlike_err for a coarse grid within parameter space
		# Make coarse grid
		G = np.array(np.meshgrid( *[np.linspace(grid_bounds[i][0],grid_bounds[i][1],grid_length) for i in range(ndim)] ))
		G = G.reshape(ndim,ndim*grid_length).T

		# Make coarse histogram of marginalized pdf
                coarse_hist_data = np.histogram(samples.T[marg_ax],range=(grid_bounds[marg_ax][0],grid_bounds[marg_ax][1]),bins=grid_length)
                coarse_dx = coarse_hist_data[1][1] - coarse_hist_data[1][0]
                coarse_xpoints = coarse_hist_data[1][1:] - coarse_dx/2.
                coarse_pdf = coarse_hist_data[0]

		## Use Cholesky Decomposition to select only cells within 2sigma of Gaussian approximation to joint posterior
		X = samples - pdf_cent
		Xcov = np.inner(X.T,X.T)/len(samples)
		L = la.cholesky(Xcov)
		invL = la.inv(L)

		# Transform the coarse grid and select those within 2sigma
		Gsph = np.dot(invL,(G-pdf_cent).T).T
		Gsph_R = np.sqrt(np.array(map(sum,Gsph**2)))
		select = np.where(Gsph_R < 2.0)[0]

		# Make prediction
		Gsph2 = np.dot(W.E.invL,(G-W.E.fid_params).T).T
		Gsph2 = Gsph2.reshape(ndim,ndim*grid_length).T
		lnlike_err_pred		= np.zeros(Gsph2.shape)
		lnlike_err_pred[select]	= GP.predict(Gsph2[select],eval_MSE=False)
		lnlike_err_pred = lnlike_err_pred.reshape([grid_step for i in range(ndim)])

		# Relate to log-posterior errors
		lnpost_err = 1*lnlike_err_pred

		# Use Eqn. (5) to get discretized, marginalized posterior error, also need to perform discrete integration
		post_err = np.zeros(grid_length)
		for i in range(ndim):
			if i == marg_ax: continue
			# Create slice array
			sl = [slice(0,grid_length) for i in range(ndim)]
			sl_p,sl_n = slice_arr[:],slice_arr[:]
			sl_p[marg_ax] = 1
			sl_n[marg_ax] = 0
			# Get delta y along this particular axes
			Delta_y = G[sl_p] - G[sl_n]


		# Make a Guess at the skewness (I've found that real skew is ~5*stats.skew()
		skew_guess = np.abs(stats.skew(xpoints)*5.)

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
		self.S.end_pos = end_pos

	def samp_drive_mpi(self,pos,step_num=500,burn_num=100,mpi_np=5,sampler_init_kwargs={},lnprob_kwargs={},sampler_kwargs={},workspace=None):
		"""
		drive sampler using mpirun
		"""
		if workspace is None:
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
                if filename is None:
                        filename = 'sampler_%s.pkl' % '_'.join(time.asctime().split(' '))
        
                if clobber == False and os.path.isfile(filename) == True:
                        print 'file exists, quitting...'

                file = open(filename,'wb')
                output = pkl.Pickler(file)
                output.dump({'S':self.S})
                file.close()


	def samp_predict_newTS(self,kd_kwargs={'bandwidth':0.2}):
		"""
		"""
		samples = self.S.sampler.chain[:, :, :].reshape((-1, self.S.ndim))
		

	def samp_common_priors_init(self,dic):
		self.CP = common_priors(dic)

	############################################
	############ Build Training Set ############
	############################################

	def TSbuild_init(self,dic):
		self.B = drive_21cmFAST(dic)


        def TSbuilder_save(self,filename,clobber=False):
                if filename is None:
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

	def plot_corner(self,levels=None):
		if levels is None:
			levels = [0.34,0.68,0.90,0.95]
		fig = corner.corner(samples.T[::-1].T, labels=p_latex[::-1], truths=p_true[::-1], range=param_bounds[::-1])



	def plot_performance(self):
		"""
		
		- Make plots that are useful for inspecting the performance of the emulator, sampler etc.
		"""
		pass












