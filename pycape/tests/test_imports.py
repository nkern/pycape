from unittest import TestCase

class TestImport(TestCase):

    def test_basic_imports(self):
        try:
            import numpy
            import astropy
            import scipy
            import sklearn
            import emcee
            self.assertGreatEqual(float(sklearn.__version__), 0.18, msg='sklearn must be >= 0.18')
        except:
            self.fail('failed basic emulator imports')