import numpy as np
import unittest
import pycape
import warnings
import py21cmsense
import os
import cPickle as pkl

class TestObs(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_21cmsense(self):
        os.chdir(py21cmsense.__path__[0])
        os.chdir('../')

        # Initialize class
        CS = py21cmsense.Calc_Sense()

        # Make an Array File
        CS.make_arrayfile('hera127', out_fname='hera127_af')

        # Calculate 1D sensitivities
        CS.calc_sense_1D('hera127_af.npz', out_fname='hera127_sense', eor='ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2') 

    def test_pspec_handling(self):
        # Load mock obs file
        file = open(pycape.__packagepath__+'/tests/mockObs_hera331_allz.pkl','rb')
        mock_data = pkl.Unpickler(file).load()
        file.close()

        # Configure data
        z_num = 44
        names = ['sense_kbins','sense_PSdata','sense_PSerrs']
        for n in names:
            mock_data[n] = np.array(mock_data[n],object)
            for i in range(z_num):
                # Cut out inf and nans
                try: mock_data[n][i] = mock_data[n][i].T[mock_data['valid'][i]].T.ravel()
                except: mock_data[n]=list(mock_data[n]);mock_data[n][i]=mock_data[n][i].T[mock_data['valid'][i]].T.ravel()
                if n == 'sense_PSerrs':
                    # Cut out sense_PSerrs / sense_PSdata > x%
                    err_thresh = 0.75        # 200%
                    small_errs = np.where(mock_data['sense_PSerrs'][i] / mock_data['sense_PSdata'][i] < err_thresh)[0]
                    mock_data['sense_kbins'][i] = mock_data['sense_kbins'][i][small_errs]
                    mock_data['sense_PSdata'][i] = mock_data['sense_PSdata'][i][small_errs]
                    mock_data['sense_PSerrs'][i] = mock_data['sense_PSerrs'][i][small_errs]

        mock_data['sense_kbins'] = np.array( map(lambda x: np.array(x,float),mock_data['sense_kbins']))

        # add ps
        model_x     = mock_data['kbins']
        obs_x       = mock_data['sense_kbins']
        obs_y       = mock_data['sense_PSdata']
        obs_y_errs  = mock_data['sense_PSerrs']
        obs_track   = np.array(map(lambda x: ['ps' for i in range(len(x))], obs_x))
        track_types = ['ps']

        # add xe
        model_x     = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(model_x,z_array)))
        obs_x       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_x,z_array)))
        obs_y       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y,np.zeros(z_num))))
        obs_y_errs  = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y_errs,np.ones(z_num)*1e6)))
        obs_track   = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_track,['xe' for i in range(z_num)])))
        track_types += ['xe']

        # add Tb
        model_x     = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(model_x,z_array)))
        obs_x       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_x,z_array)))
        obs_y       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y,np.zeros(z_num))))
        obs_y_errs  = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y_errs,np.ones(z_num)*1e6)))
        obs_track   = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_track,['Tb' for i in range(z_num)])))
        track_types += ['Tb']

        # finalize
        obs_y = np.concatenate(obs_y)
        obs_y_errs = np.concatenate(obs_y_errs)
        obs_x_nums = np.array([len(x) for x in obs_x])
        obs_track = np.concatenate(obs_track.tolist())
        track_types = np.array(track_types)

        # instantiate observation class
        O = pycape.Obs(model_x,obs_x,obs_y,obs_y_errs,obs_track,track_types)

        # update dictionary
        update_obs_dic = {'z_num':z_num}
        O.update(update_obs_dic)

        # test track
        _ = O.track(['ps'])
        _ = O.track(['ps'], arr=O.mat2row(O.x_ext) )

if __name__ == '__main__':
    unittest.main()



