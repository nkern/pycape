import unittest

class TestImport(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_numpy(self):
        import numpy

    def test_astropy(self):
        import astropy

    def test_scipy(self):
        import scipy

    def test_sklearn(self):
        import sklearn

    def test_emcee(self):
        import emcee

if __name__ == '__main__':
    unittest.main()
