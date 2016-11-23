"""
drive_pystan.py : driver for pystan

"""

import pystan

model_code = """
data {
  int<lower = 0> N;		// Number of observational data points
  int M;				// Number of polynomial weights
  real y[N];			// model prediction
  real sigma[N];		// observation error
  real xhat[M];			// polynomial design matrix coefficients
}
parameters {
  real<lower=0, upper=1.5> sigma8;
  real<lower=0, upper=1e5> Tvir;
  real<lower=0, upper=100> zeta;
}
transformed parameters {
  real mu[N];			// model prediction
  real A[M];			// design matrix
  

  mu <- xhat * A
}
model {
  y ~ normal(mu,sigma)
  sigma8 ~ normal(0.81,0.04)
}
"""        


class drive_pystan(object):

	def __init__(self,dic):
		self.__dict__.update(dic)


	def model(self,y_obs,regress_code):
		"""
		model
		"""
		# Get number of observational data points
		N = len(y_obs)








