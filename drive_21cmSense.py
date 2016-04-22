"""
Driver for 21cmSense code
See Pober et al. 2013AJ....145...65P and Pober et al. 2014ApJ...782...66P
Code Repo: https://github.com/jpober/21cmSense
"""

import os
import numpy as np

def drive_21cmSense(calib_file,ps_filename,\
		foreground_model='mod',buff=0.1,freq=0.150,ndays=180,n_per_day=6,bdwidth=0.008,nchan=82,\
		dir_21cmSense='/Users/nkern/Desktop/Research/Software/21cmSense'):
	"""
	Calculate telescope sensitivity to a 21cm power spectrum
	- calib_file should be the calibration file without '.py' ex: hera331.py => hera331
	- Note that the ps_file should have 1st column as k bins and 2nd column as power spectrum delta^2(k)
	"""

	# Use calibration file to create *.npz file
	os.system('%s/mk_array_file.py -C %s' % (dir_21cmSense,calib_file))

	# Use *.npz file to get sensitivity measurements
	os.system('python %s/calc_sense.py -m %s -b %s -f %s --eor %s --ndays %s --n_per_day %s --bdwidth %s \
		--nchan %s %s*.npz' % (dir_21cmSense,foreground_model,buff,freq,ps_filename,ndays,n_per_day,bdwidth,nchan,calib_file))

	# Load 21cm PS
	model = np.loadtxt(ps_filename)
	kbin = model[:,0]
	PS = model[:,1]

	# Load 21cmSense errors
	sense = np.load(calib_file+'drif_mod_%0.3f.npz'%freq)
	err_kbins = sense['ks']
	err_mag = sense['errs']

	valid = np.where((errs!=np.inf)&(np.isnan(errs)!=True))[0]
	err_kbins = err_kbins[valid]
	err_mag = err_mag[valid]

	return kbin, PS, err_kbins, err_mag


