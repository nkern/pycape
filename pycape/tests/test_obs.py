import numpy as np
import unittest
import pycape
import warnings
import py21cmsense
import os

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
        file = open(pycape.__path__+'/tests/mockObs_hera331_allz.pkl','rb')
        mock_data = pkl.Unpickler(file).load()
        file.close()



if __name__ == '__main__':
    unittest.main()



