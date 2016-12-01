from unittest import TestCase

class TestImport(TestCase):

    def test_one(self):
        try:
            import numpy
            import astropy
            import scipy
            import sklearn
            import emcee
            import matplotlib
        except:
            self.fail('failed imports')
