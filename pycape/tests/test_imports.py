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
        self.assertGreaterEqual(float(sklearn.__version__), 0.18, msg='sklearn must be >= 0.18')

    def test_emcee(self):
        import emcee

if __name__ == '__main__':
    unittest.main()
