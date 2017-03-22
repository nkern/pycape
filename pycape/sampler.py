"""
sampler.py

Samp class for MCMC sampling of posterior
"""
import numpy as np
import emcee
import scipy.linalg as la
import scipy.stats as stats

__all__ = ['Samp']


class Samp(object):
    """
    Sampler class

    N_params : int
        number of parameters to vary in MCMC sampling

    param_bounds : ndarray (dtype=float, shape=(N_params,2))
        hard parameter bounds for each parameter

    lnlike : function (default=None)
        A function that evalutes the log-likelihood given (theta, ydata, model, invcov)

    lnprior : function (defualt=None)
        A function that evaluates the total log-prior given (theta)

    Emu : Emu class instantiation (default=None)

    Obs : Obs class instantiation (default=None)
    """

    def __init__(self, N_params, param_bounds, lnlike=None, lnprior=None, Emu=None, Obs=None):
        """ initialize class """
        self.N_params = N_params
        self.param_bounds = param_bounds
        self.__name__ = 'Samp'

        # Check for Emu or Obs class instantiations
        if Emu is None:
            print("There is no emulator class attached to this Sampler, which will eventually be needed. \
                    See Samp.attach_class()")
            self.hasEmu = False
        else:
            self.E = Emu
            self.hasEmu = True

        if Obs is None:
            print("There is no observation class attached to this Sampler, which will eventually be needed. \
                    See Samp.attach_class()")
            self.hasObs = False
        else:
            self.O = Obs
            self.hasObs = True

        # Create lnlike
        if lnlike is not None:
            self.lnlike = lnlike

        # Create lnprior funcs
        self.lnprior_funcs = []
        # Initialize flat priors for all parameters if lnprior is none
        for i in range(self.N_params):
            if lnprior is None:
                self.lnprior_funcs.append(self.create_flat_lnprior(self.param_bounds[i],index=i,return_func=True))
            else:
                self.lnprior_funcs.append(lnprior[i])

    def attach_class(self,_class,classname=None):
        """
        Attach a different class instantiation to this class
        If it is an Emu or Obs class, it gets the name E or O

        _class : class object
            instantiation of a class object

        classname : str
            name of class instantiation (not the name of class itself)

        classtype : str
            type of class (either Emu or Obs)
        """

        # Check if it is an Emu or Obs class, which are special
        if _class.__name__ == 'Emu':
            self.E = _class
            self.hasEmu = True
        elif _class.__name__ == 'Obs':
            self.O = _class
            self.hasObs = True
        else:
            self.__dict__[classname] = _class

    def update(self,dic):
        self.__dict__.update(dic)

    def emcee_init(self, nwalkers, ndim, lnprob, ntemps=None, lnprob_kwargs={}, sampler_kwargs={}, PT=False):
        """
        Initialize an ensemble sampler

        nwalkers : int
            number of walkers

        ndim : int
            number of parameter dimensions

        lnprob : method
            posterior function call

        lnprob_kwargs : dict
            keyword arguments for lnprob function call

        sampler_kwargs : dict
            keyword arguments for emcee EnsembleSampler instatiation
        """
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.lnprob = lnprob
        self.ntemps = ntemps
        self.PT = PT

        if PT == True:
            self.sampler = emcee.PTSampler(ntemps, nwalkers, ndim, self.lnlike, self.lnprior, loglkwargs=lnprob_kwargs, **sampler_kwargs)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, kwargs=lnprob_kwargs, **sampler_kwargs)

    def construct_model(self, theta, predict_kwargs={},
                        add_overall_modeling_error=False, modeling_error=0.20,
                        add_model_cov=False, LAYG=False, k=50, use_tree=True, pool=None, vectorize=True):
        """
        Generate model prediction at walker position theta, create lnlike covariance matrix

        theta : ndarray (dtype=float, shape=[N_params,])
            row vector of a single walker position

        predict_kwargs : dict (default={})
            keyword arguments for Emu.predict function

        add_overall_modeling_error : bool (default=False)
            if True: add to existing covariance matrix predicted model_ydata times modeling_error
            if False: do nothing

        modeling_error : float (default=0.20)
            if add_overall_modeling_err is True: add model_ydata times modeling_error to cov
        """
        # Check for Emu and Obs classes
        if self.hasEmu == False:
            raise Exception("This Samp class has no Emu attached to it...")
        elif self.hasObs == False:
            raise Exception("This Samp class has no Obs attached to it...")

        # Set up iterator
        if pool is None:
            M = map
        else:
            M = pool.map

        # Chck for LAYG
        if LAYG == True:
            if theta.ndim == 1: theta = theta[np.newaxis,:]
            recon,recon_err,recon_err_cov,weights,weights_err = [],[],[],[],[]
            output = M(lambda x: self.E.predict(x, output=True, use_tree=use_tree, **predict_kwargs), theta)
            for i in range(len(output)):
                recon.append(output[i][0][0])
                recon_err.append(output[i][1][0])
                recon_err_cov.append(output[i][2][0])
                weights.append(output[i][3][0])
                weights_err.append(output[i][4][0])
            recon,recon_err,recon_err_cov = np.array(recon), np.array(recon_err), np.array(recon_err_cov)
            weights, weights_err = np.array(weights), np.array(weights_err)
            self.model_ydata = recon
            self.model_ydata_err = recon_err
            self.model_ydata_err_cov = recon_err_cov
            self.model_shape = self.model_ydata.shape
        else:
            if theta.ndim == 1: theta = theta[np.newaxis,:]
            if pool is not None or vectorize == False:
                output = M(lambda x: self.E.predict(x, output=True, **predict_kwargs), theta)
                recon,recon_err,recon_err_cov,weights,weights_err = [],[],[],[],[]
                for i in range(len(output)):
                    recon.append(output[i][0][0])
                    recon_err.append(output[i][1][0])
                    recon_err_cov.append(output[i][2][0])
                    weights.append(output[i][3][0])
                    weights_err.append(output[i][4][0])
                recon,recon_err,recon_err_cov = np.array(recon), np.array(recon_err), np.array(recon_err_cov)
                weights, weights_err = np.array(weights), np.array(weights_err)

            elif vectorize == True:
                output = self.E.predict(theta, output=True, **predict_kwargs)
                recon, recon_err, recon_err_cov, weights, weights_err = output

            self.model_shape            = recon.shape
            self.model_ydata            = recon
            self.model_ydata_err        = recon_err
            self.model_ydata_err_cov    = recon_err_cov

        self.data_cov = np.array([self.O.cov for i in range(self.model_shape[0])])
        new_cov = False

        # Add emulator error cov output at theta
        if add_model_cov == True:
            self.data_cov += self.model_ydata_err_cov
            new_cov = True

        # Add overall modeling error (21cmFAST ~ 15%)
        if add_overall_modeling_error == True:
            self.data_cov += np.array([np.eye(len(self.model_ydata[i])) * self.model_ydata[i] * modeling_error for i in range(self.model_shape[0])])
            new_cov = True

        self.data_invcov = []
        for i in range(self.model_shape[0]):
            if new_cov == True:
                try:
                    self.data_invcov.append( la.inv(self.data_cov[i]) )
                except:
                    self.data_invcov.append( self.O.invcov )
            else:
                self.data_invcov.append( self.O.invcov )

        self.data_invcov = np.array(self.data_invcov)

    def gauss_lnlike(self, ydata, model, invcov):
        """
        A typical Gaussian log-likelihood function
        """
        resid = ydata - model
        shape = resid.shape
        if len(shape) == 1:
            lnlike = -0.5 * np.dot(resid, np.dot(invcov, resid.T))
        else:
            lnlike = np.array([-0.5 * np.dot(resid[i], np.dot(invcov[i], resid[i])) for i in range(shape[0])])
        return lnlike

    def create_gauss_lnprior(self,mean,sigma,index=0,return_func=False):
        """
        Initialize a non-covarying Gaussian log-prior
        
        mean : float
            mean value

        sigma : float
            standard deviation value

        index : int (default=0)
            The index of this parameter wrt all other parameters

        return_func : bool (default=False)
            if True: return the 1D Gaussian prior function
            if False: append 1D Gaussian prior to lnprior_funcs
        """
        def gauss_lnprior(theta, mean=mean, sigma=sigma, index=index):
            """ gauss log-prior, theta is a [N_param] row vector of walker position """
            return np.log(stats.norm.pdf(theta[index],loc=mean,scale=sigma))

        if return_func == True:
            return gauss_lnprior
        else:
            self.lnprior_funcs.append(gauss_lnprior)

    def create_flat_lnprior(self,param_bounds,index=0,return_func=False):
        """
        Initialize a flat log-prior function

        param_bounds : length 2 list or tuple
            bounds of uniform prior (b1, b2)

        index : int (default=0)
            The index of this parameter wrt all other parameters

        return_func : bool (default=False)
            if True: return the flat prior function
            if False: append flat prior to lnprior_funcs

        """
        def flat_lnprior(theta,param_bounds=param_bounds,index=index):
            within = True
            if theta[index] < param_bounds[0] or theta[index] > param_bounds[1]:
                within = False
            if within == True:
                return np.log(1/(param_bounds[1]-param_bounds[0]))
            else:
                return -np.inf

        if return_func == True:
            return flat_lnprior
        else:
            return self.lnprior_funcs.append(flat_lnprior)

    def create_covarying_gauss_lnprior(self,mean,precision,index=0,return_func=False):
        """
        Create a gaussian log-prior covarying with other parameters
        
        mean : ndarray (dtype=float, shape=N_params_cov)
            mean vector for the N_params_cov covarying gaussian parameters

        precision : ndarray (dtype=float, shape=[N_params_cov,N_params_cov])
            precision matrix

        index : int (default=0)
            The index of this parameter wrt all other parameters

        return_func : bool (default=False)
            if True: return the Gaussian prior function
            if False: append Gaussian prior to lnprior_funcs
        """
        # Define multivariate gaussian variables
        cov = la.inv(precision)
        ndim = len(cov)
        lognormalize = np.log((1/np.sqrt((2*np.pi)**ndim*la.det(cov)))**(1./ndim))

        def cov_gauss_lnprior(theta, mean=mean, precision=precision, index=index, ndim=ndim, lognorm=lognormalize):
            beta = theta - mean
            chisq = np.dot(precision.T[index],beta)*beta[index]
            return lognorm + -0.5 * chisq

        if return_func == True:
            return cov_gauss_lnprior
        else:
            self.lnprior_funcs.append(cov_gauss_lnprior)

    def lnprior(self, theta, sphere=True, **kwargs):
        """
        Call the previously generated lnprior_funcs
        theta : ndarray (dtype=float, shape=[N_params,])
            row vector of walker position

        sphere : bool (default=False)
            if False: un-sphere theta
            else: nothing
        """
        ndim = theta.ndim
        if sphere == False:
            theta = np.dot(self.E.L, theta) + self.E.fid_params

        if ndim == 1:
            lnprior = 0
            for i in range(len(theta)):
                lnprior += self.lnprior_funcs[i](theta)
            return np.array([lnprior])
        else:
            lnprior = []
            for j in range(len(theta)):
                lnp = 0
                for i in range(len(theta.T)):
                    lnp += self.lnprior_funcs[i](theta[j])
                lnprior.append(lnp)
            return np.array(lnprior)

    def lnlike(self, theta, **lnlike_kwargs):
        """
        Evaluate log-likelihood
        theta : ndarray (dtype=float, shape=[N_params,])
            row vector of walker position
        """
        # Create model prediction
        self.construct_model(theta, **lnlike_kwargs)

        # Evaluate lnlike
        lnlike = self.gauss_lnlike(self.O.ydata, self.model_ydata, self.data_invcov)

        # Get rid of nan
        lnlike[np.where(np.isnan(lnlike)==True)] = -np.inf

        return lnlike

    def lnprob(self, theta, output='lnprob', **lnlike_kwargs):
        """
        Evaluate log-like and log-prior to get log-posterior (numerator of Bayes Thm.)
        theta : ndarray (dtype=float, shape=[N_params,])
            row vector of walker position

        output : string (default='lnprob')
            if 'lnlike': return log-likelihood
            elif 'lnprior': return log-prior
            else: return log-posterior
        """
        # Evaluate Likelihood
        lnlike = self.lnlike(theta,**lnlike_kwargs)

        # Evaluate lnprior
        lnprior = self.lnprior(theta, **lnlike_kwargs['predict_kwargs'])

        # Output lnprob or lnlike
        try:
            if output == 'lnlike':
                return lnlike
            elif output == 'lnprior':
                return lnprior
            else:
                return lnlike + lnprior
        except:
            return lnlike + lnprior

    def samp_drive(self, pos0, step_num=10, burn_num=0, save_progress=False, save_step=500, fname='chainhist_step'):
        """
        Drive MCMC ensemble sampler

        pos0 : ndarray (dtype=float, shape=[N_walkers, N_params])
            row vector of starting positions for all walkers

        step_num : int (default=500)
            number of steps (for each walker) to take after burn-in

        burn_num : int (default=500)
            number of steps to take for burn-in

        save_progress : bool (default=False)
            if True: output chain history every save_step number of steps

        save_step : int (default=500)
            number of steps to take before saving chain history

        fname : str (default='chainhist_step')
            filename prefix (without .pkl) for chain history
        """
        # Alter sampler run function
        if self.PT == True:
            def run_mcmc(pos0, step_num):
                p, logl, logp = self.sampler.sample(pos0,iterations=step_num)
                end_pos = self.sampler.chain[:,:,-1,:]
                return end_pos, 0, 0
            self.sampler.run_mcmc = run_mcmc

        # Run burn-in
        if burn_num > 0:
            end_pos, end_prob, end_state = self.sampler.run_mcmc(pos0,burn_num)
            #self.sampler.reset()
        else:
            end_pos = pos0

        # Run MCMC
        if save_progress == True:
            if save_step >= step_num: raise Exception("save_step must be < step_num")
            for s in np.arange(0,step_num,save_step):
                if (step_num-s) < save_step: save_step = step_num-s
                end_pos, end_prob, end_state = self.sampler.run_mcmc(end_pos,save_step)
                with open(fname+str(s+save_step)+'.pkl','wb') as f:
                    pkl.Pickler(f).dump({'chain':self.sampler.chain,'end_pos':end_pos})
        else:
            end_pos, end_prob, end_state = self.sampler.run_mcmc(end_pos,step_num)

        self.end_pos = end_pos

    def kfold_cross_validate(self,grid_tr,data_tr,use_pca=True,predict_kwargs={},
                            rando=None, kfold_Nclus=None, kfold_Nsamp=None, kwargs_tr={},
                            lnlike_kwargs={}, RandomState=1, pool=None, vectorize=True):
        """
        Cross validate sampler

        Input:
        ------
        grid_tr : ndarray

        data_tr : ndarray

        use_pca : bool (default=True)

        predict_kwargs : dict (default={})

        rando : ndarray

        kfold_Nclus : int (default=None)

        kfold_Nsamp : int (default=None)

        kwargs_tr : dict (default={})

        RandomState : int (default=1)

        Output:
        -------
        """
        # Assign random cv sets
        if rando is None:
            rando = np.array([[False]*len(data_tr) for i in range(Nclus)])
            rand_samp = np.random.choice(np.arange(len(data_tr)), replace=False, size=Nclus*Nsamp).reshape(Nclus,Nsamp)
            for i in range(Nclus): rando[i][rand_samp[i]] = True

        # Iterate over sets
        recon_grid = []
        recon_data = []
        lnlike_data = []
        lnlike_cv = []
        for i in range(kfold_Nclus):
            print "...working on kfold clus "+str(i+1)+":\n"+"-"*26
            data_tr_temp = data_tr[~rando[i]]
            grid_tr_temp = grid_tr[~rando[i]]
            # Train     
            self.E.train(data_tr_temp,grid_tr_temp,fid_data=self.E.fid_data,fid_params=self.E.fid_params,**kwargs_tr)
            # Cross Valid
            emu_lnlike, tru_lnlike = self.cross_validate(grid_tr[rando[i]], data_tr[rando[i]], lnlike_kwargs=lnlike_kwargs)
            recon_grid.extend(grid_tr[rando[i]])
            recon_data.extend(data_tr[rando[i]])
            lnlike_data.extend(tru_lnlike)
            lnlike_cv.extend(emu_lnlike)

        recon_grid = np.array(recon_grid)
        recon_data = np.array(recon_data)
        lnlike_data = np.array(lnlike_data)
        lnlike_cv = np.array(lnlike_cv)

        return lnlike_cv, lnlike_data, recon_data, recon_grid, rando

    def cross_validate(self,grid_cv,data_cv,lnlike_kwargs={}):
        """
        Cross validate against posterior distribution 

        grid_cv : ndarray (dtype=float, shape=[N_cv,N_params])
            An ndarray containing parameter space positions of CV set

        data_cv : ndarray (dtype=float, shape=[N_cv,N_data])
            An ndarray containing ydata for each CV set

        lnlike_kwargs : dict (default={})
            keyword arguments to feed Samp.lnprob()

        also_record : list (default=[])
            list of other Samp variable output to store
        """
        # Iterate over CV samples
        self.construct_model(grid_cv, **lnlike_kwargs)
        emu_lnlike = self.gauss_lnlike(self.O.ydata,self.model_ydata,self.data_invcov)
        tru_lnlike = self.gauss_lnlike(self.O.ydata,data_cv,self.data_invcov)

        return emu_lnlike, tru_lnlike

