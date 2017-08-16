"""
observations.py

Obs class for handling observational data
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
    x_num(z) = number of xdata at each redshift (ex. k-modes, dTb, x_e etc.)
    which need not be the same at each z

    model_xdata : ndarray (dtype=float object, shape=[z_num, x_num(z)])
    x-data values for the simulation model

    xdata : ndarray (dtype=float object, shape=[z_num, x_num(z)])
    x-data values for the observations

    ydata : ndarray (dtype=float object, shape=[z_num*x_num(z)])
    y-data values for the observations

    yerrs : ndarray (dtype=float object, shape=[z_num*x_num(z)])
    y-data errors for the observations

    ydata_cat : ndarray (dtype=string object, shape=[z_num*x_num(z)])
    non-rectangular ndarray of **strings** specifying the data categories
    ex: ['ps','ps','ps','Tb','xe']

    cat_types : ndarray (dtype=string,shape=[class_num])
    ndarray row vector with each unique data category
    ex: ['ps','Tb','xe']
    """

    def __init__(self,model_xdata,xdata,ydata,yerrs,ydata_cat,cat_types,p_true):
        self.__name__       = 'Obs'
        self.model_xdata    = model_xdata
        self.xdata          = xdata
        self.ydata          = ydata
        self.yerrs          = yerrs
        self.ydata_cat      = ydata_cat
        self.cat_types      = cat_types
        self.p_true         = p_true

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

    def row2mat(self,datavec,row2mat=True):
        """
        A conversion of an x or y data vector from row to matrix form or vice versa

        datavec : ndarray (dtype=object, shape=[z_num,x_num])
            a data vector of x data (ex. k-modes and/or redshifts) or y data (ex. PS bandpowers)

        row2mat : boolean (default=True)
            if True: convert from row2mat
            elif False: convert from mat to row
        """
        if row2mat == False:
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
        track = self.row2mat(track,row2mat=True)
        track = np.array(map(lambda x: x[0][x[1]] if len(x[0]) > 0 else np.array([]), zip(arr,track)))
        track_bool = self.row2mat(np.array(map(lambda x: x in catnames, self.ydata_cat)),row2mat=True)

        if mat == True:
            if return_bool == True:
                return track, track_bool
            else:
                return track
        else:
            if return_bool == True:
                return self.row2mat(track,row2mat=False), self.row2mat(track_bool,row2mat=False)
            else:
                return self.row2mat(track,row2mat=False)

