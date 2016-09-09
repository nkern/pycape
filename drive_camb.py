"""
driver for the Python wrapper of CAMB
See www.camb.info/
Mainly used to convert from base CMB parameters to derived LSS parameters (e.g. A_s <=> sigma_8)

A_s == power of the primordial curvature power spectrum (k = 0.05 Mpc^-1)

"""
import camb

class drive_camb(object):
	"""
	- Drive CAMB to get sigma8 and/or Theta_MC given base CMB parameters
	"""

	def __init__(self,dic):
		self.__dict__.update(dic)
		self.pars = camb.CAMBparams()

	def set_params(self,H0=67.5,ombh2=0.022,omch2=0.122,mnu=0.06,omk=0,tau=None,ns=0.965,As=2e-9,theta_mc=None):
		"""
		set_params(H0=67.5,ombh2=0.022,omch2=0.122,mnu=0.06,omk=0,tau=None,ns=0.965,As=2e-9,theta_mc=None)
		"""
		self.pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, cosmomc_theta=theta_mc, omk=omk, tau=tau)
		self.pars.set_dark_energy()
		self.pars.InitPower.set_params(ns=ns, As=As)
		self.pars.set_matter_power(redshifts=[0.],kmax=2.0)
		self.pars.NonLinear = camb.model.NonLinear_none
		self.results = camb.get_results(self.pars)

	@property
	def sigma8(self):
		return self.results.get_sigma8()[0]
			
	@property
	def theta_mc(self):
		return self.results.cosmomc_theta()

	@property
	def hlittle(self):
		return self.results.hubble_paramter(0)/100.0

	@property
	def As(self):
		return self.pars.primordial_power(0.05,0)

