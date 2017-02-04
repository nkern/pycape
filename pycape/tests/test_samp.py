import numpy as np
import unittest
import pycape
import scipy.stats as stats
import warnings
import os

class TestSamp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_samp(self):

        # Make Emulator
        N_data = 5
        X = np.linspace(1,10,N_data)
        y = 3.0*X + stats.norm.rvs(0,0.1,len(X))
        yerrs = np.ones(len(y))*0.1

        # Generate training set
        N_samples = 25
        data_tr = []
        grid_tr = []
        for theta in np.linspace(0.1,10,N_samples):
            data_tr.append(theta*X)
            grid_tr.append(np.array([theta]))

        data_tr = np.array(data_tr)
        grid_tr = np.array(grid_tr)

        # Instantiate
        N_modes = N_data
        variables = {'reg_meth':'gaussian','gp_kwargs':{},'N_modes':N_modes,'N_samples':N_samples,
                'scale_by_std':False,'scale_by_obs_errs':False,'lognorm':False}
        E = pycape.Emu(variables)

        E.sphere(grid_tr, save_chol=True)

        # Train
        E.fid_params = np.array([5.0])
        E.fid_data = E.fid_params[0]*X
        E.train(data_tr, grid_tr, fid_data=E.fid_data, fid_params=E.fid_params, use_pca=False, invL=E.invL)
        E.w_norm = np.ones(N_modes)
        E.recon_err_norm = np.ones(N_data)

        pred_kwargs = {'use_pca':False,'fast':True}
        _ = E.predict(np.array([3.0])[:,np.newaxis], **pred_kwargs)

        # Make Obs class
        ydata_cat = np.array(['x' for i in range(len(X.T))])
        ydata_cat_types = ['x']
        O = pycape.Obs(X,X,y,yerrs,ydata_cat,ydata_cat_types)

        N_params = 1
        param_bounds = [[0.1, 10]]
        S = pycape.Samp(N_params, param_bounds, Emu=E, Obs=O)

        sampler_kwargs = {} 
        nwalkers = 10
        ndim = 1
        lnprob_kwargs = {'predict_kwargs':pred_kwargs}
        S.emcee_init(nwalkers, ndim, S.lnprob, lnprob_kwargs=lnprob_kwargs, sampler_kwargs=sampler_kwargs)

        pos = np.array(map(lambda x: x + x*stats.norm.rvs(0,0.05,nwalkers),E.fid_params)).T

        S.samp_drive(pos,step_num=1,burn_num=0)


if __name__ == '__main__':
    unittest.main()

