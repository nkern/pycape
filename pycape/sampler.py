"""
sampler.py

Samp class for MCMC sampling of posterior
"""


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
        if lnlike is None:
            self.lnlike = self.gauss_lnlike
        else:
            self.lnlike = lnlike

        # Create lnprior funcs
        self.lnprior_funcs = []
        # Initialize flat priors for all parameters if lnprior is none
        for i in range(self.N_params):
            if lnprior is None:
                self.lnprior_funcs.append(self.flat_lnprior(self.param_bounds[i],index=i))
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

    def emcee_init(self, nwalkers, ndim, lnprob, lnprob_kwargs={}, sampler_kwargs={}):
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
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob,
                                                kwargs=lnprob_kwargs, **sampler_kwargs)

    def construct_model(self, theta, predict_kwargs={}, add_lnlike_cov=None,
                        add_overall_modeling_error=False, modeling_error=0.20, **kwargs):
        """
        Generate model prediction at walker position theta, create lnlike covariance matrix

        theta : ndarray (dtype=float, shape=[N_params,])
            row vector of a single walker position

        predict_kwargs : dict (default={})
            keyword arguments for Emu.predict function

        add_lnlike_cov : ndarray (dtype=float, shape=[N_data,N_data], default=None)
            Extra covariance matrix to add to existing covariance matrix

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

        # Emulate
        self.E.predict(theta, **predict_kwargs)
        self.model_ydata            = self.E.recon[0]
        self.model_ydata_err        = self.E.recon_err
        self.model_ydata_err_cov    = self.E.recon_err_cov
        self.data_cov = np.copy(self.O.cov)

        # Add emulator error cov output at theta
        if add_model_cov == True:
            self.data_cov += self.model_ydata_err_cov

        # Add overall modeling error (21cmFAST ~ 15%)
        if add_overall_modeling_error == True:
            self.data_cov += np.eye(len(self.model)) * self.model * modeling_error

        # Add other covariances in quadrature
        if add_lnlike_cov is not None:
            self.data_cov += add_lnlike_cov

        self.data_invcov = la.inv(self.data_cov)

    def gauss_lnlike(self,ydata,model,invcov):
        """
        A typical Gaussian log-likelihood function
        """
        resid = ydata - model
        return -0.5 * np.dot(resid.T, np.dot(invcov, resid))

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
            chisq = np.dot(precition.T[index],beta)*beta[index]
            return lognorm + -0.5 * chisq

        if return_func == True:
            return cov_gauss_lnprior
        else:
            self.lnprior_funcs.append(cov_gauss_lnprior)

    def lnprior(self,theta):
        """
        Call the previously generated lnprior_funcs
        theta : ndarray (dtype=float, shape=[N_params,])
            row vector of walker position
        """
        lnprior = 0
        for i in range(len(theta)):
            lnprior += self.lnprior_funcs[i](theta)
        return lnprior

    def lnprob(self,theta,**lnlike_kwargs):
        """
        Evaluate log-like and log-prior to get log-posterior (numerator of Bayes Thm.)
        theta : ndarray (dtype=float, shape=[N_params,])
            row vector of walker position
        """
        # Create Model Prediction
        self.construct_model(theta,**lnlike_kwargs)

        # Evaluate lnlike
        lnlike = self.lnlike(self.O.ydata,self.model_ydata,self.data_invcov) 

        # Evaluate lnprior
        lnprior = self.lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf

        return lnlike + lnprior

    def samp_drive(self, pos0, step_num=500, burn_num=500, save_progress=False, save_step=500, fname='chainhist_step'):
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
        # Run burn-in
        if burn_num > 0:
            end_pos, end_prob, end_state = self.sampler.run_mcmc(pos0,burn_num)
            self.sampler.reset()
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


    def cross_validate(self,grid_cv,data_cv,lnlike_kwargs={},also_record=[]):
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
        # Initialize lists
        grid_len = len(grid_cv)
        emu_lnlike = []
        tru_lnlike = []
        other_vars = dict(zip(also_record,[[] for ii in range(len(also_record))]))
        # Iterate over CV samples
        for i in range(grid_len):
            self.construct_model(grid_cv[i], **lnlike_kwargs)
            emu_lnl = self.lnlike(self.O.ydata,self.model,self.data_invcov)
            emu_lnlike.append(emu_lnl)

            for name in also_record:
                other_vars[name].append(self.__dict__[name])

            tru_lnl = self.lnlike(self.O.ydata,self.data_cv[i],self.data_invcov)
            tru_lnlike.append(tru_lnl)

        emu_lnlike = np.array(emu_lnlike)
        tru_lnlike = np.array(tru_lnlike)
        return emu_lnlike, tru_lnlike, other_vars



















