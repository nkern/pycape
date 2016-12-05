"""
==========
emulator.py
==========

An emulator in python
"""

# Import Modules
from __future__ import division
import os
import sys
import numpy as np
import numpy.linalg as la
import fnmatch
from .scripts.DictEZ import create as ezcreate
import itertools
import operator
import functools
from sklearn import gaussian_process
from sklearn import neighbors
import astropy.stats as astats

try: from memory_profiler import memory_usage
except: pass

__all__ = ['Emu']

class Emu(object):
    ''' perform clustering, Karhunen Loeve Transform (PCA) and interpolation on data vector "data"
    -- class klfuncs() needs to be fed a dictionary containing the relevant variables, depending on which sub-function will be used:
        N_samples   : number of samples in training set, indexed with "i"
        N_data      : number of data points in each sample, indexed with "k"
        N_modes     : number of eigenmodes, indexed with "j"
        N_params    : number of independent parameters in parameter space
    '''

    def __init__(self,dic):
        self.__name__ = 'Emu'
        self.__dict__.update(dic)
        self._trained = False

    @property
    def print_pars(self):
        keys = self.__dict__.keys()
        vals = self.__dict__.values()
        sort = np.argsort(keys)
        string = "\n" + "-"*5 + "    " + "Class Parameters" + "    " + "-"*5 + "\n" + "="*40 + "\n"
        for i in range(len(keys)):
            if type(vals[sort[i]])==np.int or type(vals[sort[i]])==np.str or type(vals[sort[i]])==np.float or type(vals[sort[i]])==np.bool:
                string += keys[sort[i]] + '\r\t\t\t = ' + str(vals[sort[i]]) + '\n'
        print string

    def update(self,dic):
        self.__dict__.update(dic)

    def param_check(self,data,params):
        try:
            if self.N_samples != data.shape[0]:
                self.N_samples = data.shape[0]
        except:
            self.N_samples = data.shape[0]
        try:
            if self.N_data != data.shape[1]:
                self.N_data = data.shape[1]
        except:
            try:
                self.N_data = data.shape[1]
            except:
                self.N_data = len(data)
        try:
            if self.N_params != params.shape[1]:
                self.N_params = params.shape[1]
        except:
            try:
                self.N_params = params.shape[1]
            except:
                self.N_params = len(params)

    def keep_file(self,filelist,string):
        '''if string in filelist, keep file'''
        return np.array(fnmatch.filter(filelist,string))

    def sphere(self,params,fid_params=None,save_chol=False,invL=None,attach=True,**kwargs):
        """
        Perform Cholesky decomposition and whiten or 'sphere' the data into non-covarying basis
        Xcov must be positive definite
        """
        if fid_params is None:
            fid_params = np.array(map(np.median,params.T))

        # Subtract mean
        X = params - fid_params

        # Find cholesky
        if invL is None:
            # Find Covariance
            Xcov = np.inner(X.T,X.T)/self.N_samples

            L = la.cholesky(Xcov)
            invL = la.inv(L)

        if save_chol == True:
            self.grid_tr = params
            self.fid_params = fid_params
            self.L = L
            self.invL = invL

        # Transform to non-covarying basis
        if attach == True:
            self.Xsph = np.dot(invL,X.T).T
        else:
            return Xsph

    def create_tree(self,data,tree_type='ball',leaf_size=50,metric='euclidean'):
        if tree_type == 'ball':
            self.tree = neighbors.BallTree(data,leaf_size=leaf_size,metric=metric)
        elif tree_type == 'kd':
            self.tree = neighbors.KDTree(data,leaf_size=leaf_size,metric=metric)

    def nearest(self, theta, k=10, use_tree=False):
        """
        Get Nearest Neighbors from sphered theta
        """
        if use_tree == True:
            if 'tree' not in self.__dict__:
                self.sphere(self.grid_tr)
                self.create_tree(self.Xsph)
            grid_D, grid_NN = self.tree.query(theta, k=k)
        else:
            R = np.array(map(la.norm, self.Xsph-theta))
            near = np.argsort(R)
            grid_D, grid_NN = self.Xsph[near][:k], near[:k]

        return grid_D, grid_NN

    def poly_design_mat(self,Xrange,dim=2,degree=6):
        """
        - Create polynomial design matrix given discrete values for dependent variables
        - dim : number of dependent variables 
        - degree : degree of polynomial to fit
        - Xrange is a list with dim # of arrays, with each array containing
            discrete values of the dependent variables that have been unraveled for dim > 1
        - Xrange has shape dim x Ndata, where Ndata is the # of discrete data points
        - A : Ndata x M design matrix, where M = (dim+degree)!/(dim! * degree!)
        - Example of A for dim = 2, degree = 2, Ndata = 3:
            A = [   [ 1  x  y  x^2  y^2  xy ]
                    [ 1  x  y  x^2  y^2  xy ]
                    [ 1  x  y  x^2  y^2  xy ]   ]
        """

        # Generate all permutations
        perms = itertools.product(range(degree+1),repeat=dim)
        perms = np.array(map(list,perms))

        # Take the sum of the powers, sort, and eliminate sums > degree
        sums = np.array(map(lambda x: reduce(operator.add,x),perms))
        argsort = np.argsort(sums)
        sums = sums[argsort]
        keep = np.where(sums <= degree)[0]
        perms = perms[argsort][keep]

        # Create design matrix
        to_the_power = lambda x,y: np.array(map(lambda z: x**z,y))
        dims = []
        for i in range(dim):
            dims.append(to_the_power(Xrange[i],perms.T[i]).T)
        dims = np.array(dims)

        A = np.array(map(lambda y: map(lambda x: functools.reduce(operator.mul,x),y),dims.T)).T

        return A

    def chi_square_min(self,y,A,N,regulate=False,fast=False):
        '''
        - perform chi square minimization
        - A is data model
        - N are weights of each y_i for fit
        - y are dataset
        '''
        # Solve for coefficients xhat
        if regulate == True:
            coeff = np.dot( np.dot(A.T,la.inv(N)), A)
            coeff_diag = np.diagonal(coeff)
            penalty = coeff_diag.mean()/1e10 * np.eye(A.shape[1])
            xhat = np.dot( la.inv(coeff + penalty), np.dot( np.dot(A.T,la.inv(N)), y) )
        else:
            xhat = np.dot( la.inv( np.dot( np.dot(A.T,la.inv(N)), A)), np.dot( np.dot(A.T,la.inv(N)), y) )

        if fast == True:
            return xhat, np.zeros(A.shape)
        else:
            # Get error
            Ashape = A.shape
            resid = (y - np.dot(A,xhat))
            sq_err = np.abs(np.dot(resid.T,resid)/(Ashape[0]-Ashape[1]))
            return xhat, np.sqrt(sq_err)

    def cluster_noise(self,basis,z):
        # Do K-Means clustering to get samples, if so KM_nclus must be defined in variables
        KM_est = KMeans(n_clusters=self.KM_nclus)
        KM_est.fit(basis)
        KM_clus = np.array(map(lambda x:np.where(KM_est.labels_==x)[0] ,range(self.KM_nclus)))
        KM_std = np.empty((self.N_samples,self.N_modes))
        for j in range(self.N_modes):
            for l in range(self.KM_nclus):
                KM_std.T[j][KM_clus[l]] = np.std(z.T[j][KM_clus[l]])
            # Make sure std is not zero
            zeros = KM_std.T[j] < 1e-10
            if len(np.where(zeros==True)[0]) > 0:
                KM_std.T[j][zeros] = np.median(KM_std.T[j][~zeros])

        names = ['KM_est','KM_clus','KM_std']
        self.update(ezcreate(names,locals()))


    def klt(self,data_tr,fid_data=None):
        ''' compute KL transform and calculate eigenvector weights for each sample in training set (TS)
            data        : [N_samples, N_data] 2D matrix, containing data of TS
            fid_data    : [N_data] row vector, containing fiducial data
            param_samples   : [N_samples, N_params] 2D matrix, containing parameter values for each sample in TS

            Necessary parameters when initializing klfuncs:
            N_modes     : scalar, number of eigenmodes to keep after truncation
            N_samples   : scalar, number of samples in training set
        '''
        
        # Compute fiducial data set
        if fid_data is None:
            fid_data = np.array(map(np.median,data_tr.T))

        # Find self-variance of mean-subtracted data
        D = (data_tr - fid_data)
        Dstd = np.array(map(astats.biweight_midvariance,D.T))

        if self.scale_by_std == True:
            if 'Dstd' not in self.__dict__.keys():
                self.Dstd = np.array(map(astats.biweight_midvariance,D.T))
            D /= self.Dstd

        if self.scale_by_obs_errs == True:
            D /= self.obs_errs

        # Find Covariance
        Dcov = np.inner(D.T,D.T)/self.N_samples

        # Solve for eigenvectors and values using SVD
        u,eig_vals,eig_vecs = la.svd(Dcov)

        # Sort by eigenvalue
        eigen_sort = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[eigen_sort]
        eig_vecs = eig_vecs[eigen_sort]

        # Solve for per-sample eigenmode weight constants
        w_tr = np.dot(D,eig_vecs.T)

        # Truncate eigenmodes to N_modes # of modes
        eig_vals        = eig_vals[:self.N_modes]
        eig_vecs        = eig_vecs[:self.N_modes]
        w_tr            = w_tr[:,:self.N_modes]
        tot_var         = sum(eig_vals)
        rec_var         = sum(eig_vals)
        frac_var        = rec_var/tot_var

        w_norm = np.array(map(lambda x: np.abs(x).max()/2,w_tr.T)).T
        w_tr /= w_norm

        # Update to Namespace
        names = ['D','Dstd','data_tr','Dcov','eig_vals','eig_vecs','w_tr','tot_var','rec_var','frac_var','fid_data','w_norm']
        self.update(ezcreate(names,locals()))

    def klt_project(self,data):
        '''
        Having already run klt() to get eigenvectors and eigenvalues, project vector 'data' onto eigenmodes
        '''
        # Subtract fiducial data from data
        D = data - self.fid_data
        if self.scale_by_std == True:
            D /= self.Dstd

        if self.scale_by_obs_errs == True:
            D /= self.obs_errs

        # Project onto eigenvectors
        self.w_tr = np.dot(D,self.eig_vecs.T)
        self.w_norm = np.array(map(lambda x: np.abs(x).max()/2,self.w_tr.T)).T
        self.w_tr /= self.w_norm


    def cross_validate(self,grid_cv,data_cv,use_pca=True,predict_kwargs={}):
        # Sphere data
        X = grid_cv - self.fid_params
        Xsph = np.dot(self.invL,X.T).T

        # Solve for eigenmode weight constants
        if use_pca == True:
            D = data_cv - self.fid_data
            if self.scale_by_std == True:
                D /= self.Dstd

            if self.scale_by_obs_errs == True:
                D /= self.obs_errs

            self.weights_true_cv = np.dot(D,self.eig_vecs.T)    

        # Predict
        self.predict(grid_cv.T,**predict_kwargs)
        self.recon_cv       = np.copy(self.recon)
        self.recon_err_cv   = np.copy(self.recon_err)
        self.weights_cv     = np.copy(self.weights)
        self.weights_err_cv = np.copy(self.weights_err)

    def train(self,data,param_samples,
            fid_data=None,fid_params=None,noise_var=None,gp_kwargs_arr=None,emode_variance_div=1.0,
            use_pca=True,compute_klt=True,calc_noise=False,norm_noise=False,verbose=False,
            group_modes=False,save_chol=False,invL=None,fast=False,pool=None,**kwargs):
        ''' fit regression model to then be used for interpolation
            noise_var   : [N_samples] row vector with noise variance for each sample in LLS solution

            Necessary parameters when initializing klfuncs:
            N_samples   : scalar, number of samples in TS
            poly_deg    : degree of polynomial to fit for
            scale_by_std    : scale data by its standard dev
            reg_meth    : method of regression, ['poly','gaussian']
            klt() variables
            gp_kwargs variables
        '''
        # Check parameters are correct
        self.param_check(data,param_samples)

        # Sphere parameter space vector
        self.sphere(param_samples,save_chol=save_chol,invL=invL,fid_params=fid_params)

        # Compute y vector
        if compute_klt == True and use_pca == True:
            self.klt(data,fid_data=fid_data)
            y = self.w_tr

        elif compute_klt == False and use_pca == True:
            self.klt_project(data)
            y = self.w_tr

        elif use_pca == False:
            y = data

        # Calculate noise estimates
        if calc_noise == True:
            self.cluster_noise(self.Xsph, y)

        # polynomial regression
        if self.reg_meth == 'poly':
            # Compute design matrix
            param_samp_ravel = map(list,self.Xsph.T)
            A = self.poly_design_mat(param_samp_ravel,dim=self.N_params,degree=self.poly_deg)

            # Use LLS over training set to solve for weight function polynomials
            if noise_var is None:
                noise_var = np.array([2]*self.N_samples*self.N_modes)           # all training set samples w/ equal weight
                noise_var = noise_var.reshape(self.N_samples,self.N_modes)

            if calc_noise == True:
                noise_var = self.KM_std**2

            if norm_noise == True:
                noise_var /= y**2

            # Fill weight matrix
            W = np.zeros((self.N_modes,self.N_samples,self.N_samples))
            for i in range(self.N_modes):
                if noise_var is None:
                    np.fill_diagonal(W[i],1e-5)
                elif noise_var is not None:
                    if type(noise_var) == float or type(noise_var) == int:
                        np.fill_diagonal(W[i],noise_var)    
                    else:
                        np.fill_diagonal(W[i],noise_var[i])

            # LLS for interpolation polynomial coefficients
            xhat = []
            stand_err = []
            for i in range(self.N_modes):
                if fast == True:
                    xh = self.chi_square_min(y.T[i],A,W[i],fast=fast)
                    err = 0.0
                else:
                    xh, err = self.chi_square_min(y.T[i],A,W[i],fast=fast)
                xhat.append(xh)
                stand_err.append(err)
            xhat = np.array(xhat).T
            stand_err = np.array(stand_err).T

        # Gaussian Process Regression
        elif self.reg_meth == 'gaussian':
            # Initialize GP, fit to data
            GP = []
            if verbose == True: print '...Eigenmode group variance divider = ',emode_variance_div
            for j in range(self.N_modegroups):
                if verbose == True:
                    print '...Working on eigenmode #'+str(j)
                    print '-'*40

                if gp_kwargs_arr is None:
                    gp_kwargs = self.gp_kwargs.copy()
                else:
                    gp_kwargs = gp_kwargs_arr[j].copy()

                if calc_noise == True:
                    gp_kwargs['alpha'] = self.KM_std.T[j]

                if norm_noise == True:
                    if 'alpha' not in gp_kwargs: gp_kwargs['alpha'] = 1e-1
                    gp_kwargs['alpha'] = (gp_kwargs['alpha']/y.T[self.modegroups[j][0]])**2

                # Create GP
                gp = gaussian_process.GaussianProcessRegressor(**gp_kwargs)
                GP.append(gp)

            # Fit GPs!
            # Use parallel processing for hyperparameter regression (optional)
            if pool is None:
                M = map
            else:
                M = pool.map

            M(lambda x: x[0].fit(self.Xsph,y.T[self.modegroups[x[1]]].T), zip(GP,np.arange(len(GP))))
            GP = np.array(GP)
            if pool is not None:
                pool.close()

        # Update to namespace
        self._trained = True
        names = ['xhat','stand_err','GP']
        self.update(ezcreate(names,locals()))

    def GPhyperParam_NN(self,k=10,kwargs_tr={}):
        self.E.GPhyperParams = []
        for i in range(len(self.E.grid_tr)):
            parsph = np.dot(self.E.invL,self.E.grid_tr[i])
            grid_NN = self.E.tree.query(parsph,k=k)[1][0]
            self.E.N_samples = len(grid_NN)
            self.emu_train(self.E.data_tr[grid_NN],self.E.grid_tr[grid_NN],fid_params=self.E.fid_params,fid_data=self.E.fid_data,kwargs_tr=kwargs_tr)
            if solve_simul == True:
                W.E.GPhyperParams.append(self.E.GP.theta_)
            else:
                W.E.GPhyperParams.append(map(lambda x: x.theta_,self.E.GP))
        self.E.GPhyperParams = np.array(self.E.GPhyperParams)

    def group_eigenmodes(self,emode_variance_div=10.0):
        '''
        - group eigenmodes together based on eigenvalue
        '''
        if emode_variance_div == 1.0:
            self.N_modegroups = self.N_modes
            self.modegroups = np.arange(self.N_modes)[:,np.newaxis]
        elif emode_variance_div > 1e8:
            self.N_modegroups = 1
            self.modegroups = np.arange(self.N_modes)[np.newaxis,:]
        else:   
            eigval_max = self.eig_vals[0]
            eigval_min = self.eig_vals[-1]
            dividers = 10**(np.arange(np.log10(eigval_max),np.log10(eigval_min)+1e-8,-np.log10(emode_variance_div))[1:])
            N_dividers = len(dividers) + 1
            dividers = np.array([np.inf]+list(dividers)+[0])
            modegroups = []
            for i in range(N_dividers):
                select = np.where((self.eig_vals>=dividers[i+1])&(self.eig_vals<dividers[i]))[0]
                if len(select)==0: continue
                modegroups.append(select)
            modegroups=np.array(modegroups)
            N_modegroups = len(modegroups)

            self.N_modegroups = N_modegroups
            self.modegroups = modegroups

    def predict(self,Xpred,use_Nmodes=None,GPs=None,fast=False,\
        use_pca=True,group_modes=False,**kwargs):
        '''
        - param_vals is ndarray with shape [N_params,N_samples]

        - transform param_vals to Sphered_Param_Vals
        - calculate eigenmode construction of predicted signal given param_vals
        eigenmode = weights * eigenvector = w_tr * f_j
        - given a list of parameter vals (ex. [[0.85],[40000.0],[30.0]]) and assuming eigenmodes
        and the best-fit polynomial of their weights have been trained over a training set,
        calculate the weight constants (w_tr) of each eigenmode
        '''
        # Chi Square Multiplier, 95% prob
        self.csm = np.sqrt([3.841,5.991,7.815,9.488,11.070,12.592,14.067,15.507,16.919])

        # Transform to whitened parameter space
        Xpred_shape = Xpred.shape
        Xpred_sph = np.dot(self.invL,(Xpred-self.fid_params).T).T

        if use_Nmodes is None:
            use_Nmodes = self.N_modes

        # Polynomial Interpolation
        if self.reg_meth == 'poly':
            # Calculate weights
            if Xpred_sph.ndim == 1: Xpred_sph = Xpred_sph.reshape(1,len(Xpred_sph))
            A = self.poly_design_mat(Xpred_sph.T,dim=self.N_params,degree=self.poly_deg)
            weights = np.dot(A,self.xhat)

            # Renormalize weights
            weights *= self.w_norm

            # Compute construction
            if use_pca == True:
                recon = np.dot(weights.T[:use_Nmodes].T,self.eig_vecs[:use_Nmodes])
            else:
                recon = weights.T[0]

            if self.scale_by_std == True:
                recon *= self.Dstd

            if self.scale_by_obs_errs == True:
                recon *= self.obs_errs

            recon += self.fid_data

            weights_err = self.stand_err.reshape(weights.shape) * self.w_norm
 
        # Gaussian Process Interpolation
        if self.reg_meth == 'gaussian':
            weights,MSE = [],[]
            iterate = self.N_modegroups
            # Iterate over GPs
            for j in range(iterate):
                if GPs != None:
                    if fast == True:
                        w = GPs[j].predict(Xpred_sph,return_cov=False)
                        weights.extend(w.T)
                        MSE.extend(np.zeros(w.shape).T)
                    else:
                        w,mse = GPs[j].predict(Xpred_sph,return_cov=True)
                        mse = np.sqrt(mse.diagonal())
                        weights.extend(w.T)
                        MSE.extend(np.ones(w.shape).T*mse)
                else:
                    if fast == True:
                        w = self.GP[j].predict(Xpred_sph,return_cov=False)
                        weights.extend(w.T)
                        MSE.extend(np.zeros(w.shape).T)
                    else:
                        w,mse = self.GP[j].predict(Xpred_sph,return_cov=True)
                        mse = np.sqrt(mse.diagonal())
                        weights.extend(w.T)
                        MSE.extend(np.ones(w.shape).T*mse)

            weights,MSE = np.array(weights).T,np.array(MSE).T
            if weights.ndim == 1:
                weights = weights.reshape(Xpred_shape[1],len(weights))
                MSE = MSE.reshape(Xpred_shape[1],len(weights))

            # Renormalize weights
            weights *= self.w_norm
            weights_err = np.sqrt(MSE) * np.sqrt(self.w_norm)

            # Compute reconstruction
            if use_pca == True:
                recon = np.dot(weights.T[:use_Nmodes].T,self.eig_vecs[:use_Nmodes])
            else:
                recon = weights.T[0]

            if self.scale_by_std == True:
                recon *= self.Dstd

            if self.scale_by_obs_errs == True:
                recon  *= self.obs_errs

            recon += self.fid_data

        # Calculate Error
        if use_pca == True:
            emode_err = np.array(map(lambda x: (x*self.eig_vecs.T).T, weights_err))
            recon_err = np.sqrt( np.array(map(lambda x: np.sum(x,axis=0),emode_err**2)) )
            recon_err_cov = np.array([[np.outer(self.eig_vecs[j],self.eig_vecs[j])*weights_err[i][j]**2 for j in range(self.N_modes)] for i in range(len(recon))])
            recon_err_cov = np.sum(recon_err_cov, axis=1)

        else:
            recon_err = weights_err

        # Normalize Error
        recon_err *= self.recon_err_norm
        recon_err_cov *= self.recon_err_norm**2

        # Construct data product and error on data product
        if fast == False:
            names = ['recon','weights','MSE','weights_err','Xpred_sph','recon_err','recon_err_cov']
            self.update(ezcreate(names,locals()))
        else:
            self.recon = recon
            self.recon_err = recon_err
            self.weights = weights
            self.weights_err = weights_err
            self.recon_err_cov = recon_err_cov

    # Check if regression of a certain eigenmode is really working via cross validation
    def check_cross_valid(self):
        pass




