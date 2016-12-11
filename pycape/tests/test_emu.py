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
        variables = {'reg_meth':'gaussian','gp_kwargs':{},'N_modes':1,'N_samples':len(X),
                    'scale_by_std':False}
        E = Emu(variables)

        E.sphere(X, save_chol=True)

        # Train
        E.train(y, X, use_pca=False)
        E.w_norm = np.ones(100)

        # Predict
        Xp = np.array(np.meshgrid(*[np.linspace(-3,3,10),np.linspace(-3,3,10)])).reshape(2,100).T
        yp = E.predict(Xp, use_pca=False)

