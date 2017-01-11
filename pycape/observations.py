"""
observations.py

Obs class for handling observational data, including
a driver for the 21cmSense code

For more on 21cmSense, see:
Parsons et al. 2012ApJ...753...81P
Pober et al. 2013AJ....145...65P
Pober et al. 2014ApJ...782...66P
Code Repo: https://github.com/jpober/21cmSense
"""
import os
import numpy as np
import scipy.linalg as la
import cPickle as pkl
import warnings
import operator

try:
    from py21cmsense import Calc_Sense
except ImportError:
    warnings.warn('\nCould not import py21cmsense')

__all__ = ['Obs']

class Obs(object):
    """
    Observation class
    Define:
    z_num = number of redshifts
    x_num(x) = number of xdata at each redshift (ex. k-modes, dTb, x_e etc.)
    which need not be the same at each z

    model_xdata : ndarray (dtype=float object, shape=[z_num, x_num(z)])
    x-data values for the simulation model

    xdata : ndarray (dtype=float object, shape=[z_num,x_num(z)])
    x-data values for the observations

    ydata : ndarray (dtype=float object, shape=[z_num,x_num(z)])
    y-data values for the observations

    yerrs : ndarray (dtype=float object, shape=[z_num,x_num(z)])
    y-data errors for the observations

    ydata_cat : ndarray (dtype=string object, shape=[z_num,x_num(z)])
    non-rectangular ndarray of **strings** specifying the data categories
    ex: ['ps','ps','ps','Tb','xe']

    cat_types : ndarray (dtype=string,shape=[class_num])
    ndarray row vector with each unique data category
    ex: ['ps','Tb','xe']
    """

    def __init__(self,model_xdata,xdata,ydata,yerrs,ydata_cat,cat_types):
        self.__name__       = 'Obs'
        self.model_xdata    = model_xdata
        self.xdata          = xdata
        self.ydata          = ydata
        self.yerrs          = yerrs
        self.ydata_cat      = ydata_cat
        self.cat_types      = cat_types

        try:
            self.x_ext          = np.concatenate(list(self.xdata))
        except:
            self.x_ext          = self.xdata
        self.N_data         = len(self.x_ext)
        self.cov            = np.eye(self.N_data)*self.yerrs**2
        self.invcov         = la.inv(self.cov)

    def update(self,dic):
        """
        update
        """
        self.__dict__.update(dic)

    def var_check(self,var):
        """
        A checker to see if necessary variables have been instatiated inside class namespace

        var : list of strings
        """
        for v in var:
            if v not in self.__dict__.keys():
                raise Exception("'"+v+"' not defined in Obs namespace...")

    def update_cov(self,cov_add):
        """
        upate cov
        """
        self.cov    += cov_add
        self.invcov = la.inv(self.cov)

    def interp_PSk_mod2obs(self,model_ydata):
        """
        An interpolation from model power spectra k-bins to observational k-bins
        
        model_ydata : ndarray (dtype=object, shape=[z_num,k_num])
            an ndarray containing power spectra k-modes of model
        """
        # Interpolate model basis onto observation basis
        model_interp = []
        for i in range(len(model_ydata)):
            model_interp.append(np.interp(self.xdata[i],self.model_xdata[i],model_ydata[i]) )

        return np.array(model_interp)

    def mat2row(self,datavec,mat2row=False):
        """
        A conversion of an x or y data vector from matrix to row form or vice versa

        datavec : ndarray (dtype=object, shape=[z_num,x_num])
            a data vector of x data (ex. k-modes and/or redshifts) or y data (ex. PS bandpowers)

        mat2row : boolean (default=False)
            if True: convert from mat2row
            elif False: convert from row2mat
        """
        if mat2row == True:
            try:
                datavec = np.concatenate(datavec.tolist())
            except ValueError:
                datavec_temp = []
                for i in range(len(datavec)):
                    datavec_temp.extend(list(datavec[i]))
                datavec = np.array(datavec_temp)
            return datavec

        else:
            datavec = list(datavec)
            datavec2 = []
            for i in range(len(self.xdata)):
                datavec2.append(np.array([datavec.pop(0) for j in range(len(self.xdata[i]))]))

            return np.array(datavec2)

    def track(self,catnames,arr=None,mat=True,return_bool=False):
        """
        Track or isolate a data category from self.ydata_cat

        catnames : ndarray (dtype=string, shape=[cat_num,]
            a list of data categories you'd like to track and isolate

        arr : ndarray (dtype=object, shape=[z_num,x_num(z)])
            ndarray of x-data or y-data form from which you'd like to isolate data of a particular category

        mat : bool (default=True)
            if True: return in matrix form
            else: return in row vector form

        return_bool : bool (default=False)
            if True: return a boolean ndarray of shape self.xdata with catnames equal to True
        """
        # check for arr
        if arr is None: arr = self.xdata
        
        # get categories
        track = reduce(operator.add,map(lambda x: self.ydata_cat==x,catnames))
        track = self.mat2row(track,mat2row=False)
        track = np.array(map(lambda x: x[0][x[1]] if len(x[0]) > 0 else np.array([]), zip(arr,track)))
        track_bool = self.mat2row(np.array(map(lambda x: x in catnames, self.ydata_cat)),mat2row=False)

        if mat == True:
            if return_bool == True:
                return track, track_bool
            else:
                return track
        else:
            if return_bool == True:
                return self.mat2row(track,mat2row=True), self.mat2row(track_bool,mat2row=True)
            else:
                return self.mat2row(track,mat2row=True)

	def drive_21cmSense(self,calib_file,ps_filenames,
                        data_filename=None,obs_direc=None,write_direc=None,write_data=True,
			            foreground_model='mod',buff=[0.1],freq=[0.135],ndays=180,n_per_day=6,
                        bwidth=[0.008],nchan=[82],lowk_cut=0.15):
		"""
		Calculate telescope sensitivity to a 21cm power spectrum
		Calib_file should be the calibration file without '.py' ex: hera331.py => hera331
		Note that the ps_file should have 1st column as k bins and 2nd column as power spectrum delta^2(k)
		Keyword Arguments : 
		data_filename=None,obs_direc=None,write_direc=None,write_data=True
		foreground_model='mod',buff=[0.1],freq=[0.135],ndays=180
		n_per_day=6,bwidth=[0.008],nchan=[82],lowk_cut=0.15
		"""
		# Set parameters
		if obs_direc == None:
			obs_direc = self.dir_pycape+'/ObsData'

		if write_direc == None:
			write_direc = '.'

		if data_filename == None:
			data_filename = 'mock_21cmObs.pkl'

        # Instantiate Calc_Sense
        CS = Calc_Sense(**kwargs)

		# Use calibration file to create *.npz file
        os.system('python %s/mk_array_file.py -C %s' % (self.dir_21cmSense,calib_file))

		# Move *.npz file to proper directory
        os.system('mv %s*.npz %s/' % (calib_file,obs_direc))

		# Configure data arrays
        kbins = []
        PSdata = []
        sense_kbins = []
        sense_PSdata = []
        sense_PSerrs = []	

        valid = []
        # Use *.npz file to get sensitivity measurements
        len_files = len(ps_filenames)
        for i in range(len_files):
            print ''
            print 'working on file: '+ps_filenames[i]
            print '-'*30
            os.system('python %s/calc_sense.py -m %s -b %s -f %s --eor %s --ndays %s --n_per_day %s --bwidth %s \
                --nchan %s %s/%s.drift_blmin*.npz' % (self.dir_21cmSense,foreground_model,buff[i],freq[i],ps_filenames[i],
                ndays,n_per_day,bwidth[i],nchan[i],obs_direc,calib_file))

            # Move *.npz file to proper directory
            os.system('mv %s*.npz %s/' % (calib_file,obs_direc))

            # Load 21cm PS
            model = np.loadtxt(ps_filenames[i])
            kb = model[:,0]
            PSdat = model[:,1]

            # Load 21cmSense errors
            sense = np.load(obs_direc+'/'+calib_file+'.drift_mod_%0.3f.npz'%freq[i])
            sense_kb = sense['ks']
            sense_PSerr = sense['errs']

            valid.append( (sense_PSerr!=np.inf)&(np.isnan(sense_PSerr)!=True)&(sense_kb>lowk_cut) )

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
        self.kbins = kbins
        self.PSdata = PSdata
        self.sense_kbins = sense_kbins
        self.sense_PSdata = sense_PSdata
        self.sense_PSerrs = sense_PSerrs
        self.valid = valid
        self.freq = freq

        # Write to file
        if write_data == True:
            file = open(write_direc+'/'+data_filename,'wb')
            output = pkl.Pickler(file)
            output.dump(data_dic)
            file.close()

