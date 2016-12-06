import numpy as np
from pycape import Emu
import unittest

class TestEmu(unittest.TestCase):

    def __init__(self):
        pass

    def test_simple(self):
        """
        Simple Emulator Test
        """
        fail = False
        X = np.array(map(lambda x: x.ravel(), np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))))
        y = np.sinc(np.sqrt(X[0]**2 + X[1]**2))

        X = X.T
        y = y[:,np.newaxis]

        # Instantiate
        variables = {'reg_meth':'gaussian','gp_kwargs':{},'N_modes':1}
        E = Emu(variables)

        # Train
        E.train(y, X, use_pca=False)

        # Predict
        yp = np.vstack([(np.random.random(10)-0.5)*10, (np.random.random(10)-0.5)*10]).T


