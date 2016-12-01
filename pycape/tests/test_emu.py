import numpy as np
from pycape import Emu
import unittest

class TestEmu(unittest.TestCase):

    def test_simple(self):
        try:
            X = np.array(map(lambda x: x.ravel(), np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))))
            y = np.sinc(np.sqrt(X[0]**2 + X[1]**2))

            X = X.T
            y = y[:,np.newaxis]

            # Instantiate
            variables = {'reg_meth':'gaussian','gp_kwargs':{}}
            E = Emu(variables)

            return
            # Train
            E.train(y, X, use_pca=False)

            # Predict
            yp = np.vstack([(np.random.random(10)-0.5)*10, (np.random.random(10)-0.5)*10]).T
        except:
            self.fail('Failed test_simple')


