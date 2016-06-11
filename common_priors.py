"""
common_priors.py : a list of common priors on physical parameters relevant to Cosmic Dawn and EoR studies

"""

def planck_priors(param_name,param_val):
	"""
	planck_priors(param_name,param_val)
	- recent planck priors on CMB parameters
	- log prior is log Gaussian
	"""
	omega_b_hh		= 0.02222
	omega_b_hh_err		= 0.00023

	omega_c_hh		= 0.1197
	omega_c_hh_err		= 0.0022

	theta_MC_hundred	= 1.04085
	theta_MC_hundred_err	= 0.00047

	n_s			= 0.9655
	n_s_err			= 0.0062

	ln_tenten_A_s		= 3.089
	ln_tenten_A_s_err	= 0.036

	tau			= 0.078
	tau_err			= 0.019

	param_names = ['omega_b_hh','omega_c_hh']





