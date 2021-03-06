"""
simulations.py

Drivers for relevant simulations
"""
import os
import warnings
from collections import OrderedDict
import numpy as np

try:
    import camb
except:
    warnings.warn('\nCould not import camb')

__all__ = ['Drive_Camb']

class Drive_Camb(object):
    """
    - Drive CAMB to get sigma8 <=> A_s and/or H0 <=> Theta_MC mappings given base CMB parameters
    """

    def __init__(self):
        self.pars = camb.CAMBparams()
        self.reion_pars = camb.ReionizationParams()

    def set_params(self,H0=67.27,ombh2=0.02225,omch2=0.1198,omk=0,ns=0.9645,As=2.2065e-9,theta_mc=None,tau=0.079,
                        omb=None, omc=None, ln10As=None):
        """
        Set cosmological parameters
        Default values consistent with Planck 2016 Parameter Constraints with TT, TE, EE + lowP

        Input:
        ------

        H0 : float, default=67.27
            Hubble Constant, km/sec/Mpc

        ombh2 : float, default = 0.02225
            Omega baryon * (H0/100)**2
            Fractional energy density of baryons w.r.t. critical density at z=0

        omch2 : float, default = 0.1198
            Omega cold dark matter * (H0/100)**2
            Fractional energy density of CDM w.r.t. critical density at z=0

        omk : float, default = 0.0
            Omega curvature

        ns : float, default = 0.9645
            Spectral tilt of primordial curvature fluctuation power spectrum

        As : float, default = 2.2065e-9
            Amplitude of primordial scalar perturbation fluctuations

        theta_mc : float, default = None
            Sound horizon at recombination, as approximated by CosmoMC

        tau : float, default = 0.079
            Electron scattering optical depth to the surface of last scattering

        ln10As : float, default = None
            Value for log_e(1e10 * As), if defined will override value of As

        Notes:
        ------
        ombh2 and omb, omch2 and omc cannot simultaneously be set.
        If omb or omc are defined, will default to using
        omb and omc values multiplied by (H0/100.0)**2
        """
        if omb is not None:
            ombh2 = omb * (H0/100.0)**2
        if omc is not None:
            omch2 = omc * (H0/100.0)**2
        if ln10As is not None:
            As = np.exp(ln10As) / 1e10

        # Set camb parameters
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
    def ln10As(self):
        return np.log(1e10*self.pars.primordial_power(0.05,0))

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

    def get_cl(self):
        self.D_ell = self.results.get_cmb_power_spectra(params=self.pars, spectra=['total'],CMB_unit='muK')['total']





