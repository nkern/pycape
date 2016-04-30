"""
Driver for 21cmSense code
See Pober et al. 2013AJ....145...65P and Pober et al. 2014ApJ...782...66P
Code Repo: https://github.com/jpober/21cmSense
"""
import os
import numpy as np
from .DictEZ import create as ezcreate
from fits_table import fits_table
import cPickle as pkl

class drive_21cmSense():

	def __init__(self,dic):
		self.__dict__.update(dic)

	def update(self,dic):
		self.__dict__.update(dic)

	def calc_sense(self,calib_file,ps_filenames,data_filename=None,obs_direc=None,write_direc=None,write_data=True,
			foreground_model='mod',buff=0.1,freq=0.135,ndays=180,n_per_day=6,bwidth=0.008,nchan=82):
		"""
		(calib_file,ps_filenames,obs_direc=None,write_direc=None)
		Calculate telescope sensitivity to a 21cm power spectrum
		- calib_file should be the calibration file without '.py' ex: hera331.py => hera331
		- Note that the ps_file should have 1st column as k bins and 2nd column as power spectrum delta^2(k)
		"""
		# Set parameters
		if obs_direc == None:
			obs_direc = self.dir_pycape+'/ObsData'

		if write_direc == None:
			write_direc = '.'

		if data_filename == None:
			data_filename = 'mock_21cmObs.pkl'

		# Use calibration file to create *.npz file
		os.system('%s/mk_array_file.py -C %s' % (self.dir_21cmSense,calib_file))

		# Move *.npz file to proper directory
		os.system('mv %s*.npz %s/' % (calib_file,obs_direc))

		# Configure data arrays
		kbins = []
		PSdata = []
		sense_kbins = []
		sense_PSdata = []
		sense_PSerrs = []	

		valid = np.array([True]*52,bool)
		self.valid = valid
		# Use *.npz file to get sensitivity measurements
		len_files = len(ps_filenames)
		for i in range(len_files):
			os.system('python %s/calc_sense.py -m %s -b %s -f %s --eor %s --ndays %s --n_per_day %s --bwidth %s \
				--nchan %s %s/%s*.npz' % (self.dir_21cmSense,foreground_model,buff,freq[i],ps_filenames[i],ndays,n_per_day,bwidth,nchan,obs_direc,calib_file))

			# Move *.npz file to proper directory
			os.system('mv %s*.npz %s/' % (calib_file,obs_direc))

			# Load 21cm PS
			model = np.loadtxt(ps_filenames[i])
			kb = model[:,0]
			PSdat = model[:,1]

			# Load 21cmSense errors
			sense = np.load(obs_direc+'/'+calib_file+'.drift_mod_%0.3f.npz'%freq)
			sense_kb = sense['ks']
			sense_PSerr = sense['errs']

			local_valid = (sense_PSerr!=np.inf)&(np.isnan(sense_PSerr)!=True)
			self.local_valid = local_valid
			if i == 0 and len_files != len(local_valid):
				valid = np.array([True]*len(local_valid),bool)

			valid *= local_valid

			sense_kb = sense_kb
			sense_PSerr = sense_PSerr

			# Interpolate between ps_file to get ps at sense_kbins
			sense_PSdat = np.interp(sense_kb,kb,PSdat)

			# Append to arrays
			kbins.append(kb)
			PSdata.append(PSdat)
			sense_kbins.append(sense_kb)
			sense_PSdata.append(sense_PSdat)
			sense_PSerrs.append(sense_PSerr)	

		kbins		= np.array(kbins)
		PSdata		= np.array(PSdata)
		sense_kbins	= np.array(sense_kbins)
		sense_PSdata	= np.array(sense_PSdata)
		sense_PSerrs	= np.array(sense_PSerrs)
		valid		= np.array(valid)

		# Append to namespace
		names = ['kbins','PSdata','sense_kbins','sense_PSdata','sense_PSerrs','valid']
		data_dic = ezcreate(names,locals())
                self.update(data_dic)

		# Write to file
		if write_data == True:
			file = open(write_direc+'/'+data_filename,'wb')
			output = pkl.Pickler(file)
			output.dump(data_dic)
			file.close()



