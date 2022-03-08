# -*- coding: utf-8 -*-
"""
VansyngelBayesianInference

Use MCMC methods to perform Bayesian inference on the Vansyngel model
parameters in the BICEP/Keck footprint by fitting to the best-fit dust
model from BICEP/Keck X.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"
__all__ = ["VansyngelBayesianInference"]


from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
import time
import logging
from typing import Optional

import dill
import scipy.stats as st
import incredible as cr
from pygtc import plotGTC
import healpy as hp
import numpy as np
import pysm
import matplotlib.pyplot as plt
import emcee

from utils import set_logger
from vansyngel_model import VansyngelModel


@dataclass
class VansyngelBayesianInference:
    """Bayesian inference of the Vansyngel Model Parameters in the
    BICEP/Keck footprint and evaluation of the MCMC fit 
    """
    
    def __init__(self, dust_map: np.ndarray, mask: np.ndarray,
                 out_dir: str) -> None:
        """Define necessary variables based on the input arguments and
        make necessary directories

        Parameters:
            dust_map (np.ndarray): dust intensity, e.g. Planck 353 GHz
            mask (np.ndarray): BICEP/Keck mask
            out_dir (str): directory to save log file, plots, and
                MCMC traces
        """
        self.dust_map = dust_map
        self.nside = hp.get_nside(dust_map)
        
        if hp.get_nside(mask) != self.nside:
            mask = hp.ud_grade(mask, self.nside)
        self.mask = mask
        
        assert type(out_dir) is str, "Output directory must be a str"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir

        set_logger(os.path.join(out_dir, "van_inference_log.log"))
        logging.info("Output directory: {}".format(out_dir))

        # Storage for MCMC samples from fitting the model
        self.samples = None
        self.Nsamples = 0
        
        self.param_names = ["p0", "alphaM", "fM"]
        # plot labels
        self.param_labels = [r"$p_0$", r"$\alpha_M$", r"$f_M$"]
        
        self.priors = {"p0":None, "alphaM":None, "fM":None}

        return None
  
    def log_prior(self, **params: float) -> float:
        """Uniform log prior PDF p(params|H)

        Parameters:
            params (float kwargs): Vansyngel model parameters to fit

        Returns:
            (float): uniform log prior value
        """
        return np.sum(st.uniform.logpdf(
            [params["p0"], params["alphaM"], params["fM"]],
            loc=[0, -5, 0], scale=[0.5, 4, 2]))

    def draw_samples_from_prior(self, Ns: int) -> list[dict]:
        """Draw samples from the prior PDF as a list of dictionaries

        Parameters:
            Ns (int): Number of samples to draw

        Returns:
            (list[dict]): list of dictionaries of random samples
                of the parameters from their uniform priors
        """
        samples = []
        for i in range(Ns):
            self.priors["p0"] = st.uniform.rvs(loc=0, scale=0.5)
            self.priors["alphaM"] = st.uniform.rvs(loc=-5, scale=4)
            self.priors["fM"] = st.uniform.rvs(loc=0, scale=2)
            samples.append(self.priors.copy())

        return samples

    def log_likelihood(self, **params: float) -> float:
        """log likelihood PDF p(y|params,H) using each bin in multipole
        as a normal distribution around the BICEP/Keck (2015) best-fit
        dust model values

        Parameters:
            params (float kwargs): dictionary of model parameters

        Returns:
            (float): normal log likelihood value
        """
        d_ell = np.mean(self.generate_replica_dataset(7, **params), axis = 0)

        # BICEP/Keck 2015 mean and standard deviation values for the
        # best-fit dust model in BB over multipole bins
        bk_exp = np.array([0.1674, 0.1272, 0.1016, 0.0870, 0.0773, 
                           0.0706, 0.0652, 0.0605, 0.0566])
        bk_std = np.array([0.1796, 0.0850, 0.0707, 0.0845, 0.1108,
                           0.1442, 0.1822, 0.2480, 0.3606])

        return np.sum(st.norm.logpdf(d_ell, loc=bk_exp, scale=bk_std))
        
    def generate_replica_dataset(self, Ns: int, **params: float) -> np.ndarray:
        """Generate a dust spectrum from the sampling distribution
        
        Parameters:
            Ns (int): Number of spectra to generate
            params (float kwargs): dictionary of model parameters

        Returns:
            (np.ndarray): dust spectra
        """
        van_model = VansyngelModel(self.nside, self.dust_map,
                                   self.mask, **params)

        d_ell = []
        for i in range(Ns):
            d_ell.append(van_model.make_dustsim())
            
        return np.array(d_ell)
        
    def log_posterior(self, parameterlist: Optional[list[float]] = None,
                      **params: float) -> float:
        """log of the unnormalized posterior PDF p(params|y,H)

        Parameters:
            parameterlist (Optional[list[float]]): for compatibility
                with emcee
            params (dict): dictionary of model parameters

        Returns:
            (float): log posterior value
        """
        if parameterlist is not None:
            pdict = {k:parameterlist[i] for i, k in enumerate(
                self.param_names)}
            return self.log_posterior(**pdict)

        lnp = self.log_prior(**params)
        if lnp != -np.inf:
            lnp += self.log_likelihood(**params)

        return lnp

    def draw_samples_from_posterior(self, starting_params: list[float],
                                    nsteps: int, nwalkers: int = 6) -> None:
        """Use emcee to draw samples from P(params|y,H)

        Parameters:
            starting_params (list[float]): initial guesses
            nsteps (int): Number of MCMC steps to run for each walker
            nwalkers (int): Number of walkers (use >= twice # of 
                parameters)
        """
        npars = len(starting_params)   

        # Generate an ensemble of walkers within +/-1% of the guess:
        theta_0 = np.array([starting_params * (
            1.0 + 0.01*np.random.randn(npars)) for j in range(nwalkers)])
        # Note that the initial parameter array theta_0 should have
        # dimensions nwalkers × npars
        
        # Use all cores available
        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(nwalkers, npars,
                                                 self.log_posterior, pool=pool)
            # Evolve the ensemble:
            self.sampler.run_mcmc(theta_0, nsteps, progress=True)
        
        # Plot the raw samples
        plt.rcParams["figure.figsize"] = (12.0, 4.0 * npars)
        fig, ax = plt.subplots(npars, 1);
        cr.plot_traces(self.sampler.chain[:min(8, nwalkers), :, :], ax,
                       labels = self.param_labels);
        plt.savefig(os.path.join(self.out_dir, "trace_plots.png"))

        return None

    def check_chains(self, burn: int, maxlag: int) -> None:
        """Ignore burn-in samples at the start of each chain and
        compute convergence criteria and effective number of samples.

        Parameters:
            burn (int): number of samples to ignore at the front of each
                chain.
            maxlag (int): a guess at the maximum lag in chain
                autocorrelation since doing the calculation to
                arbitrarily long lags becomes very expensive.
        """
        nwalk, nsteps, npars = self.sampler.chain.shape
        if burn < 1 or burn >= nsteps:
            return

        tmp_samples = [self.sampler.chain[i, burn:, :] for i in range(nwalk)]

        logging.info("R =", cr.GelmanRubinR(tmp_samples))
        logging.info("neff =",
                     cr.effective_samples(tmp_samples, maxlag=maxlag))
        logging.info("NB: Since walkers are not independent, "
                     "these will be optimistic!")

        return None

    def remove_burnin(self, burn: int) -> None:
        """Remove burn-in samples at the start of each chain and
        concatenate. Plot and store the result in self.samples

        Parameters:
            burn (int): number of samples to remove at the front of each
                chain.
        """
        nwalk, nsteps, npars = self.sampler.chain.shape
        if burn < 1 or burn >= nsteps:
            return

        self.samples = self.sampler.chain[:, burn:, :].reshape(
            nwalk * (nsteps-burn), npars)
        self.Nsamples = self.samples.shape[0]

        plt.rcParams["figure.figsize"] = (12.0, 4.0 * npars)
        fig, ax = plt.subplots(npars, 1);
        cr.plot_traces(self.samples, ax, labels=self.param_labels);
        plt.savefig(os.path.join(self.out_dir, "trace_plots_noburnin.png"))

        return None
        
    def posterior_mean(self) -> dict:
        """Compute the posterior mean of each parameter from
        MCMC samples.

        Returns:
            (dict): posterior mean of each parameter from MCMC samples
        """
        m = np.mean(self.samples, axis=0)
        return {k:m[i] for i, k in enumerate(self.param_names)}
    
    def run(self) -> None:
        """Draw samples from the posterior, check the chains, remove
        burn-in, and make plots.
        """
        nwalkers = cpu_count()
        # more accurate than time.time()
        start = time.perf_counter()
        # Carry out the parameter inference and display the Markov chains
        self.draw_samples_from_posterior(starting_params=[0.26, -2.5, 0.9],
                                         nsteps=1000, nwalkers=nwalkers * 2)
        logging.info("emcee took {0:.1f} seconds".format(
            time.perf_counter() - start))
        
        # Compute the convergence criteria and effective number of samples
        self.check_chains(burn=100, maxlag=500)
    
        # Remove burn-in for good and plot the concatenation of what's left
        self.remove_burnin(burn=100)
    
        # Check the covariances of the parameters
        plotGTC(self.samples, paramNames=self.param_labels, figureSize=6,
                customLabelFont={"size": 12}, customTickFont={"size": 12},
                customLegendFont={"size": 16})
        plt.savefig(os.path.join(self.out_dir, "param_cov.png"))
    
        logging.info("Posterior mean parameters = ", self.posterior_mean())
    
        # Save session
        dill.dump_session(
            os.path.join(self.out_dir, "fitted_Vansyngel_model.db"))

        return None


if __name__ == "__main__":
    nside = 256

    # Load Planck 353 GHz intensity map
    dust_map = hp.read_map("/home/groups/clkuo/planck_maps_for_george/pr3/353"
                           "/HFI_SkyMap_353_2048_R3.01_full.fits", 0)
    dust_map = hp.ud_grade(dust_map, nside)
    dust_map *= 1e6 # These maps are given in K_{CMB}
    dust_map = hp.Rotator(coord=["G", "C"]).rotate_map_pixel(dust_map)
    
    # BICEP/Keck mask
    mask = hp.read_map("/home/groups/clkuo/planck_maps_for_george/masks/"
                       "bk18_mask_smallfield_cel_n0512.fits")
    mask = hp.ud_grade(mask, nside)
    mask = np.nan_to_num(mask)
    mask[mask < 0] = 0

    out_dir = "/home/groups/clkuo/van_out"

    van_infer = VansyngelBayesianInference(dust_map, mask, out_dir)
    van_infer.run()
