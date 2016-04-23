"""
likelihood.py : Functions for evaluating the 21cm Likelihood

"""

class like():

	def __init__(self,dic):
		self.__dict__.update(dic)


	def gauss_like(self,data,model,covar,log=True):
		resid = data - model
		if log == True:
			return -0.5 * np.dot( resid.T, np.dot(la.inv(covar), resid) )
		else:
			return np.exp( -0.5 * np.dot( resid.T, np.dot(la.inv(covar), resid) ) )



