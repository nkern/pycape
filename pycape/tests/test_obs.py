import numpy as np
import unittest
from pycape import Obs
import warnings
import py21cmsense

class TestObs(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_21cmsense(self):
       
        # Initialize class
        CS = py21cmsense.Calc_Sense()

        # Make an Array File
        CS.make_arrayfile(py21cmsense.__path__[0]+'/hera127', out_fname='hera127_af')

        # Calculate 1D sensitivities
        CS.calc_sense_1D('hera127_af.npz', out_fname='hera127_sense', eor=py21cmsense.__path__[0]+'/ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2') 



if __name__ == '__main__':
    unittest.main()



