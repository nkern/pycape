from unittest import TestCase

class TestOne(TestCase):

    def test_import(self):
        try:
            import numpy
        except:
            self.fail('failed importing numpy')


    def test_import2(self):
        try:
            import scipy
        except:
            self.fail('failed scipy')


    def test_import3(self):
        try:
            import astropy
        except:
            self.fail('failed import')


    def test_import4(self):
        try:
            import sklearn
        except:
            self.fail('failed import')
