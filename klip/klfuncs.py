"""
==========
klfuncs.py
==========

- Functions to perform KL transform, interpolation and clustering

-------------
Nicholas Kern
2016
"""

# Import Modules
import os, sys
import numpy as np, numpy.linalg as la
import fnmatch
from .DictEZ import create as ezcreate
import itertools
import operator
import functools
from sklearn import gaussian_process
from sklearn.cluster import KMeans
from sklearn import neighbors

try: from memory_profiler import memory_usage
except: pass

class klfuncs():
	''' perform clustering, Karhunen Loeve Transform (PCA) and interpolation on data vector "data"
	-- class klfuncs() needs to be fed a dictionary containing the relevant variables, depending on which sub-function will be used:
		N_samples	: number of samples in training set, indexed with "i"
		N_data		: number of data points in each sample, indexed with "k"
		N_modes		: number of eigenmodes, indexed with "j"
		N_params	: number of independent parameters in parameter space
	'''

	def __init__(self,dic):
		self.__dict__.update(dic)

	def update(self,dic):
		self.__dict__.update(dic)

	def param_check(self,data,params):
		if self.N_samples != data.shape[0]:
			raise Exception('self.N_samples != data.shape[0]')
		if self.N_data != data.shape[1]:
			raise Exception('self.N_data != data.shape[1]')
		if self.N_params != params.shape[1]:
			raise Exception('self.N_params != params.shape[1]')

        def keep_file(self,filelist,string):
                '''if string in filelist, keep file'''
                return np.array(fnmatch.filter(filelist,string))

	def sphere(self,params,fid_params=None):
		"""
		Perform Cholesky decomposition and whiten or 'sphere' the data into non-covarying basis
		Xcov must be positive definite
		"""
		if fid_params == None:
			fid_params = np.array(map(np.median,params.T))

		# Subtract mean
		X = params - fid_params

		# Find Covariance
		Xcov = np.inner(X.T,X.T)/self.N_samples

		L = la.cholesky(Xcov)
		invL = la.inv(L)
		Xsph = np.dot(invL,X.T).T

                # Update to Namespace
		names = ['Xsph','X','Xcov','L','invL','fid_params']
                self.update(ezcreate(names,locals()))


	def create_tree(self,tree_type='ball',leaf_size=20,metric='euclidean'):
		if tree_type == 'ball':
			self.tree = neightbors.BallTree(self.Xsph,leaf_size=leaf_size,metric=metric)
		elif tree_type == 'kd':
			self.tree = neighbors.KDTree(self.Xsph,leaf_size=leaf_size,metric=metric)

	def poly_design_mat(self,Xrange,dim=2,degree=6):
		"""
		- Create polynomial design matrix given discrete values for dependent variables
		- dim : number of dependent variables 
		- degree : degree of polynomial to fit
		- Xrange is a list with dim # of arrays, with each array containing
		    discrete values of the dependent variables that have been unraveled for dim > 1
		- Xrange has shape dim x Ndata, where Ndata is the # of discrete data points
		- A : Ndata x M design matrix, where M = (dim+degree)!/(dim! * degree!)
		- Example of A for dim = 2, degree = 2, Ndata = 3:
		    A = [   [ 1  x  y  x^2  y^2  xy ]
			    	[ 1  x  y  x^2  y^2  xy ]
			    	[ 1  x  y  x^2  y^2  xy ]   ]
		- add regularization to penalize large parameter fits
		"""

		# Generate all permutations
		perms = itertools.product(range(degree+1),repeat=dim)
		perms = np.array(map(list,perms))

		# Take the sum of the powers, sort, and eliminate sums > degree
		sums = np.array(map(lambda x: reduce(operator.add,x),perms))
		argsort = np.argsort(sums)
		sums = sums[argsort]
		keep = np.where(sums <= degree)[0]
		perms = perms[argsort][keep]

		# Create design matrix
		to_the_power = lambda x,y: np.array(map(lambda z: x**z,y))
		dims = []
		for i in range(dim):
		    dims.append(to_the_power(Xrange[i],perms.T[i]).T)
		dims = np.array(dims)

		A = np.array(map(lambda y: map(lambda x: functools.reduce(operator.mul,x),y),dims.T)).T

		return A

	def chi_square_min(self,y,A,N,regulate=False):
		'''
		- perform chi square minimization
		- A is data model
		- N are weights of each y_i for fit
		- y are dataset
		'''
		if regulate == True:
			coeff = np.dot( np.dot(A.T,la.inv(N)), A)
			coeff_diag = np.diagonal(coeff)
			penalty = coeff_diag.mean()/1e10 * np.eye(A.shape[1])
			xhat = np.dot( la.inv(coeff + penalty), np.dot( np.dot(A.T,la.inv(N)), y) )
		else:
			xhat = np.dot( la.inv( np.dot( np.dot(A.T,la.inv(N)), A)), np.dot( np.dot(A.T,la.inv(N)), y) )
		return xhat

	def get_err(self,rec_data,true_data):
		'''
		- calculate rms error 
		- true_data has shape [N_samples,N_data]
		- rec_data has shape  [N_samples,N_data]
		'''
		frac = (rec_data-true_data)/true_data
		rms = np.array(map(np.median,frac))
		return rms

	def cluster_noise(self,basis,z):
		# Do K-Means clustering to get samples, if so KM_nclus must be defined in variables
		KM_est = KMeans(n_clusters=self.KM_nclus)
		KM_est.fit(basis)
		KM_clus = np.array(map(lambda x:np.where(KM_est.labels_==x)[0] ,range(self.KM_nclus)))
		KM_std = np.empty((self.N_samples,self.N_modes))
		for j in range(self.N_modes):
			for l in range(self.KM_nclus):
				KM_std.T[j][KM_clus[l]] = np.std(z.T[j][KM_clus[l]])
			# Make sure std is not zero
			zeros = KM_std.T[j] < 1e-10
			if len(np.where(zeros==True)[0]) > 0:
				KM_std.T[j][zeros] = np.median(KM_std.T[j][~zeros])

		names = ['KM_est','KM_clus','KM_std']
		self.update(ezcreate(names,locals()))


	def klt(self,data,fid_data=None,scale_by_std=False):
		''' compute KL transform and calculate eigenvector weights for each sample in training set (TS)
			data		: [N_samples, N_data] 2D matrix, containing data of TS
			fid_data	: [N_data] row vector, containing fiducial data
			param_samples	: [N_samples, N_params] 2D matrix, containing parameter values for each sample in TS

			Necessary parameters when initializing klfuncs:
			N_modes		: scalar, number of eigenmodes to keep after truncation
			N_samples	: scalar, number of samples in training set
		'''
		
		# Compute fiducial data set
		if fid_data == None:
			fid_data = np.array(map(np.median,data.T))

		# Find self-variance of mean-subtracted data
		D = (data - fid_data)
		Dstd = np.array(map(np.std,D.T))

		if scale_by_std == True:
			D /= Dstd

		# Find Covariance
		Dcov = np.inner(D.T,D.T)/self.N_samples

		# Solve for eigenvectors and values using SVD
		u,eig_vals,eig_vecs = la.svd(Dcov)

		# Sort by eigenvalue
		eigen_sort = np.argsort(eig_vals)[::-1]
		eig_vals = eig_vals[eigen_sort]
		eig_vecs = eig_vecs[eigen_sort]

		# Solve for per-sample eigenmode weight constants
		a_ij = np.dot(D,eig_vecs.T)

		# Truncate eigenmodes to N_modes # of modes
		tot_var         = sum(eig_vals)
		eig_vals        = eig_vals[:self.N_modes]
		eig_vecs        = eig_vecs[:self.N_modes]
		a_ij            = a_ij[:,:self.N_modes]
		rec_var         = sum(eig_vals)
		frac_var        = rec_var/tot_var

		# Update to Namespace
		names = ['D','Dcov','eig_vals','eig_vecs','a_ij','tot_var','rec_var','frac_var','fid_data']
		self.update(ezcreate(names,locals()))

	def cross_validate(self,data_cv,grid_cv,fid_data=None,fid_params=None):
		# Sphere data
		if fid_params == None:
			fid_params = np.array(map(np.median,grid_cv.T))
		X = grid_cv - fid_params
		Xsph = np.dot(self.invL,X.T).T

                # Solve for eigenmode weight constants
                if fid_data == None:
                        fid_data = np.array(map(np.median,data_cv.T))

		D = data_cv - fid_data
                self.a_ij_cv = np.dot(D,self.eig_vecs.T)	

		# Predict
		self.calc_eigenmodes(grid_cv.T)
		self.weights_cv		= np.copy(self.weights)
		self.eig_modes_cv	= np.copy(self.eig_modes)
		self.MSE_cv		= np.copy(self.MSE)
		self.conf90_cv		= np.copy(self.conf90)
		self.recon_cv		= np.copy(self.recon)
		self.par_sph_cv		= np.copy(self.par_sph)

	def klinterp(self,data,param_samples,\
			fid_data=None,fid_params=None,sample_noise=None,\
			scale_by_std=False,calc_noise=False):
		''' compute KL transform and interpolate KL eigenmode weights over parameter space.
		 	sample_noise	: [N_samples] row vector with noise for each sample in LLS solution

			Necessary parameters when initializing klfuncs:
			N_samples	: scalar, number of samples in TS
			poly_deg	: degree of polynomial to fit for
			scale_by_std	: scale data by its standard dev
			reg_meth	: method of regression, ['poly','gaussian']
			klt() variables
			gp_kwargs variables
		'''
		# Check parameters are correct
		self.param_check(data,param_samples)

		# Sphere parameter space vector
		self.sphere(param_samples,fid_params=fid_params)

		# Compute KLT
		self.klt(data,fid_data=fid_data,scale_by_std=scale_by_std)

		# Calculate noise estimates
		basis = self.Xsph
		if calc_noise == True:
			self.cluster_noise(basis,self.a_ij)

		# polynomial regression
		if self.reg_meth == 'poly':
			# Compute design matrix
			param_samp_ravel = map(list,basis.T)
			A = self.poly_design_mat(param_samp_ravel,dim=self.N_params,degree=self.poly_deg)

			# Use LLS over training set to solve for weight function polynomials
			if sample_noise == None:
				sample_noise = np.array([1]*self.N_samples*self.N_modes)           # all training set samples w/ equal weight
				sample_noise = sample_noise.reshape(self.N_samples,self.N_modes)
			elif calc_noise == True:
				sample_noise = self.KM_std

			# Fill weight matrix
			W = np.zeros((self.N_modes,self.N_samples,self.N_samples))
			for i in range(self.N_modes):
				np.fill_diagonal(W[i],sample_noise[i])
				
			# LLS for interpolation polynomial coefficients
			xhat = []
			for i in range(self.N_modes):
				xhat.append(self.chi_square_min(self.a_ij.T[i],A,W[i]))
			xhat = np.array(xhat).T

		# Gaussian Process Regression
		elif self.reg_meth == 'gaussian':
			# Initialize GP, fit to data
			GP = []
			for j in range(self.N_modes):
				print '...Working on eigenmode #'+str(j)
				print '-'*40
				gp_kwargs = dict(zip(self.gp_kwargs.keys(),self.gp_kwargs.values()))

				if sample_noise == None:
					gp_kwargs.update({'nugget':1e-5})
				elif sample_noise != None:
					gp_kwargs.update({'nugget':(sample_noise.T[j]/self.a_ij.T[j])**2})
				elif calc_noise == True:
					gp_kwargs.update({'nugget':(self.KM_std.T[j]/self.a_ij.T[j])**2})

				# Fit!
				gp = gaussian_process.GaussianProcess(**gp_kwargs).fit(basis,self.a_ij.T[j])

				GP.append(gp)
			GP = np.array(GP)

		else:
			raise Exception("regression method '"+str(reg_meth)+"' not understood")

		# Update to namespace
		names = ['xhat','GP','KM_est','KM_clus','KM_std']
		self.update(ezcreate(names,locals()))

	def calc_eigenmodes(self,param_vals,use_Nmodes=None,GPs=None,return_global=True):
                '''
		- param_vals is ndarray with shape [N_params,N_samples]

		- transform param_vals to Sphered_Param_Vals
                - calculate eigenmode construction of predicted signal given param_vals
                        eigenmode = weights * eigenvector = a_ij * f_j
                - given a list of parameter vals (ex. [[0.85],[40000.0],[30.0]]) and assuming eigenmodes
                        and the best-fit polynomial of their weights have been trained over a training set,
                        calculate the weight constants (a_ij) of each eigenmode
                '''
		# Chi Square Multiplier, 95% prob
		self.csm = np.sqrt([3.841,5.991,7.815,9.488,11.070,12.592,14.067,15.507,16.919,\
				18.307,19.675,21.026,22.362,23.685,24.996,26.296,27.587,28.869,30.144,31.410])

		# Transform to whitened parameter space
		param_vals = (param_vals.T - self.fid_params).T
		par_sph = np.dot(self.invL,param_vals)

		# Polynomial Interpolation
		if self.reg_meth == 'poly':
	                # Calculate weights
	                A = self.poly_design_mat(par_sph,dim=self.N_params,degree=self.poly_deg)
	                weights = np.dot(A,self.xhat)

	                # Compute eigenmodes in data space
	                eig_modes = []
			for l in range(weights.shape[0]):
				eig_modes.append(np.array(weights[l]*self.eig_vecs.T).T)
			eig_modes=np.array(eig_modes)

		# Gaussian Process Interpolation
		if self.reg_meth == 'gaussian':

			# Get prediction
			weights,MSE,conf90 = [],[],[]
			for j in range(self.N_modes):
				if GPs != None:
					w,mse = GPs[j].predict(par_sph.T,eval_MSE=True)
				else:
					w,mse = self.GP[j].predict(par_sph.T,eval_MSE=True)
				weights.append(w)
				MSE.append(mse)
				conf90.append(np.sqrt(mse)*self.csm[0])

			weights,MSE,conf90 = np.array(weights).T,np.array(MSE).T,np.array(conf90).T

			# Compute eigenmodes
			self.weights = weights
			eig_modes = []
			for l in range(weights.shape[0]):
				eig_modes.append(np.array(weights[l]*self.eig_vecs.T).T)
			eig_modes=np.array(eig_modes)

		# Construct data product
		if use_Nmodes == None:
			recon = self.fid_data + np.array(map(lambda x: reduce(operator.add,x), eig_modes))
		else:
			recon = self.fid_data + np.array(map(lambda x: reduce(operator.add,x[:use_Nmodes]), eig_modes))

		names = ['recon','eig_modes','weights','MSE','conf90','par_sph']
                self.update(ezcreate(names,locals()))







