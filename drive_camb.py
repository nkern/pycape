"""
driver for the Python wrapper of CAMB
See www.camb.info/

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
		
		self.pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, cosmomc_theta=theta_mc, omk=omk, tau=tau)
		self.pars.set_dark_energy()
		self.pars.InitPower.set_params(ns=ns, As=As)
		self.pars.set_matter_power(reshifts=0.,kmax=2.0)
		self.pars.NonLinear = camb.model.NonLinear_none
		self.results = camb.get_results(self.pars)

	@property
	def sigma8(self):
		return camb.get_results(self.pars).get_sigma8()
			
	@property
	def theta_mc(self):
		return camb.get_results(self.pars).cosmomc_theta()
