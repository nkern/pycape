import numpy as np
import pycape
import unittest
import warnings
import scipy.stats as stats

class TestEmu(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_no_pca(self):
        """
        Simple Emulator Test Using no PCA
        """
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
                'scale_by_std':False,'scale_by_obs_errs':False}
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

if __name__ == '__main__':
    unittest.main()
