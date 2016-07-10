"""
drive_pymc3.py : driver for pymc3, specifically using its capabilities for HMC and NUTS

Use the analytic mapping from simulation inputs to simulation outputs created by the emulator
to get gradient information of the likelihood function, which can then inform a Hamiltonian Monte Carlo
sampling method.

Currently only polynomial emulators (not Gaussian Processes) are supported.

--------------
Nicholas Kern
2016
"""

import numpy as np
import pymc3 as pm
import itertools
import operator
import functools
import scipy.optimize as optimize

class drive_pymc3(object):

	def __init__(self,dic):
		self.__dict__.update(dic)

	def polynomial_design_matrix(self,params,degree=1,dim=1):
		"""
		polynomial_design_matrix(self,params,degree=1,dim=1)

		- Create a polynomial design matrix with dim dimensions (free dependent parameters) with order degree.
		- Output is a *string*, which when used with the eval() function will create the design matrix

		params : 1d array or list of length dim, containing names of free parameters as a string
				example: params = ['alpha','beta','sigma']

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
		to_the_power = lambda x,y: np.array(map(lambda z: x+'**'+str(z),y))
		coeffs = []
		for i in range(dim):
			coeffs.append(to_the_power(params[i],perms.T[i]).T)

		coeffs = np.array(coeffs)

		A = np.array(map(lambda x: '('+')*('.join(x)+')',coeffs.T)).T
		return A


	def define_model(self,params,priors,likelihood=None,Y=[1],sigma=[1]):

		# Get number of parameters
		N = len(params)

		# Initialize model container
		self.basic_model = pm.Model()

		with self.basic_model:

			# Assign priors on free dependent parameters
			for i in range(N):
				self.__dict__[params[i]] = eval(priors[i])

			#  Create Polynomial Design Matrix
			self.S.A = self.polynomial_design_matrix(params,degree=self.degree,dim=self.dim)

			# Combine it with the polynomial weights solved for by the emulator to get model prediction
			mu = eval( ' + '.join(map(lambda x:'*'.join(np.round(x,6)),zip(self.E.xhat,self.S.A))) )

			# Define likelihood
			if likelihood is None:
				Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
			else:
				Y_obs = eval(likelihood)


		def get_MAP(self,fmin=None):
			print '...getting MAP'
			if fmin is None:
				self.map_estimate = pm.find_MAP(model=self.basic_model)
			else:
				self.map_estimate = pm.find_MAP(model=self.basic_model,fmin=fmin)

		def HMC(self,Nsteps=100):

			with self.basic_model:

				# Check for map
				if 'map_estimate' not in self.__dict__:
					self.get_MAP()

				# Initialize sampler
				step = pm.NUTS(scaling=self.map_estimate)

				# Draw samples
				self.trace = pm.sample(Nsteps, step, start=self.map_estimate)






