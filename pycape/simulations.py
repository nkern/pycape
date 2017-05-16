"""
simulations.py

Drivers for relevant simulations
"""
import os
import warnings
from collections import OrderedDict

try:
    import camb
except:
    warnings.warn('\nCould not import camb')

__all__ = ['Drive_Camb','Drive_21cmFAST']

class Drive_Camb(object):
    """
    - Drive CAMB to get sigma8 <=> A_s and/or H0 <=> Theta_MC mappings given base CMB parameters
    """

    def __init__(self):
        self.pars = camb.CAMBparams()
        self.reion_pars = camb.ReionizationParams()

    def set_params(self,H0=67.5,ombh2=0.022,omch2=0.122,mnu=0.06,omk=0,tau=None,ns=0.965,As=2e-9,theta_mc=None):
        """
        set_params(H0=67.5,ombh2=0.022,omch2=0.122,mnu=0.06,omk=0,tau=None,ns=0.965,As=2e-9,theta_mc=None)
        """
        self.ns = ns
        self.pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, cosmomc_theta=theta_mc, omk=omk, tau=tau)
        self.pars.set_dark_energy()
        self.pars.InitPower.set_params(ns=ns, As=As)
        self.pars.set_matter_power(redshifts=[0.],kmax=2.0)
        self.pars.NonLinear = camb.model.NonLinear_none
        self.results = camb.get_results(self.pars)

    @property
    def sigma8(self):
        return self.results.get_sigma8()[0]
			
    @property
    def theta_mc(self):
        return self.results.cosmomc_theta()

    @property
    def hlittle(self):
        return self.pars.H0/100.0

    @property
    def As(self):
        return self.pars.primordial_power(0.05,0)

    @property
    def ombh2(self):
        return self.pars.omegab * (self.hlittle)**2

    @property
    def omch2(self):
        return self.pars.omegac * (self.hlittle)**2

    @property
    def tau(self):
        return self.pars.Reion.optical_depth

    @property
    def get_pars(self):
        return OrderedDict(zip(['sigma8','theta_mc','hlittle','As','ombh2','omch2','ns','tau'],\
                    [self.sigma8,self.theta_mc,self.hlittle,self.As,self.ombh2,self.omch2,self.ns,self.tau]))


