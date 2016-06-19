"""
common_priors.py : a list of common priors on physical parameters relevant to Cosmic Dawn and EoR studies

For CMB priors see
Planck Collaboration (1502.01589)
Planck Collaboration (1605.03507)

"""
from collections import OrderedDict

cmb_priors1 = OrderedDict{
'name'				: 'Planck TT,TE,EE+lowP',
'H0'                : 67.27,
'H0_err'            : 0.66,
'ombh2'				: 0.02225,
'ombh2_err'			: 0.00016,
'omch2'				: 0.1198,
'omch2_err'			: 0.0015,
'theta_mc100'		: 1.04077,
'theta_mc100_err'	: 0.00032,
'ns'				: 0.9645,
'ns_err'			: 0.0049,
'lntentenAs'		: 3.094,
'lntentenAs_err'	: 0.034,
'tau'			    : 0.079,
'tau_err'		    : 0.017,
'sigma8'            : 0.831,
'sigma8_err'        : 0.013,
'zre'				: 10.0,
'zre_err'			: 1.6}

cmb_priors2 = OrderedDict{
'name'              : 'Planck TT,TE,EE+lowP+lensing+ext',
'H0'                : 67.74,
'H0_err'            : 0.46,
'ombh2'             : 0.02230,
'ombh2_err'         : 0.00014,
'omch2'             : 0.1188,
'omch2_err'         : 0.0010,
'theta_mc100'       : 1.04093,
'theta_mc100_err'   : 0.00030,
'ns'                : 0.9667,
'ns_err'            : 0.0040,
'lntentenAs'        : 3.064,
'lntentenAs_err'    : 0.023,
'tau'               : 0.066,
'tau_err'           : 0.012,
'sigma8'            : 0.8159,
'sigma8_err'        : 0.0086,
'zre'               : 8.8,
'zre_err'           : 1.2}

cmb_priors3 = {
'name'			: 'Planck TT+lollipop',
'ombh2'                 : 0.02222,
'ombh2_err'             : 0.00023,
'omch2'                 : 0.1205,
'omch2_err'             : 0.0022,
'theta_mc100'           : 1.04085,
'theta_mc100_err'       : 0.00047,
'ns'                    : 0.9620,
'ns_err'                : 0.0050,
'lntentenAs'            : 3.059,
'lntentenAs_err'        : 0.020,
'tau'                   : 0.058,
'tau_err'               : 0.012}






