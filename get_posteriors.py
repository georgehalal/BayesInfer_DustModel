import sys
sys.path.append("~/.local/lib/python3.6/site-packages")
import healpy as hp
import numpy as np
import pylab as pl
import pysm
import matplotlib.pyplot as plt
import scipy.stats as st
import emcee
import incredible as cr
from pygtc import plotGTC
from multiprocessing import Pool,cpu_count
import dill
import time


def do_projection(B, Bdims = 'sp'):
    """Project the magnetic field onto a sphere

    Args:
        B: magnetic field
        Bdims: dimensions of magnetic field
    """
    B = B / np.sqrt(np.einsum('%s,%s->p'%(Bdims, Bdims), B, B))
    Bperp = B - r * np.einsum('%s,sp->p'%Bdims, B, r)
    phi = np.pi - np.sign(np.einsum('sp,sp->p', Bperp, e)) * np.arccos(np.einsum('sp,sp->p', \
        Bperp, n) / np.sqrt(np.einsum('sp,sp->p', Bperp, Bperp)))
    cos2gamma = 1. - np.abs(np.einsum('%s,sp->p'%Bdims, B, r))**2

    return phi, cos2gamma


def simulate_GMF(l0, b0, p0, alphaM, fM, N):
    """Simulate the Galactic magnetic field as sum of mean and turbulent components
        B = B0 + fM Bt

    Args:
        l0 and b0: Galactic longitude and latitude defining B0 orientation
        p0: effective polarization fraction of dust emission
        alphaM: power-law index of correlated-Gaussian field power spectrum for Bt
        fM: relative strength of the random component of the field
        N: number of shells approximating integral along line-of-sight
    """
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    N = int(N)

    B0 = np.array([np.cos(l0) * np.cos(b0), np.sin(l0) * np.cos(b0), np.sin(b0)])
    
    phi = np.zeros((N, npix))
    cos2gamma = np.zeros((N, npix))
    
    Cell = ell**alphaM
    Cell[:2] = 0.
    
    for i in range(N):
        while True:
            Bt = np.array([hp.synfast(Cell, nside, verbose = False), hp.synfast(Cell, nside, \
                verbose = False), hp.synfast(Cell, nside, verbose = False)])
            Bt = Bt / np.sqrt(np.einsum('sp,sp->p', Bt, Bt))
            B = B0[:,np.newaxis] + fM * Bt
            phi[i], cos2gamma[i] = do_projection(B)
            if (np.sum(np.isnan(phi[i])) + np.sum(np.isnan(cos2gamma[i]))) == 0: break
                
    return cos2gamma, phi


def make_dustsim(**params):
    """Make a realization of dust polarization power spectrum

    Args:
        params: dictionary of model parameters to fit
    """
    dust_sim = np.zeros((3, npix))
    
    cos2gamma, phi = simulate_GMF(70*d2r, 24*d2r, params['p0'], params['alphaM'], params['fM'], 7)
    
    S = dust_map / np.sum(1. - (cos2gamma - 2. / 3.) * params['p0'], axis=0)
    
    # Use Planck 353 GHz as dust intensity
    dust_sim[0] = dust_map
    dust_sim[1] = S * params['p0'] * np.sum(np.cos(2. * phi) * cos2gamma, axis=0)
    dust_sim[2] = S * params['p0'] * np.sum(np.sin(2. * phi) * cos2gamma, axis=0)
    
    # Use PySM to scale to 220 GHz
    sky = pysm.Sky(nside = nside, preset_strings = ["d0"])
    sky.components[0].I_ref = dust_sim[0] * pysm.units.uK_CMB
    sky.components[0].Q_ref = dust_sim[1] * pysm.units.uK_CMB
    sky.components[0].U_ref = dust_sim[2] * pysm.units.uK_CMB
    sky.components[0].freq_ref_P = sky.components[0].freq_ref_I
    model_freq_maps = sky.get_emission(220 * pysm.units.GHz)
    
    Dell_V = ell * (ell + 1) * hp.anafast(mask * model_freq_maps, lmax = lmax)[2] / (fsky * 2 * np.pi)
    
    # Use BICEP/Keck binning
    DV = np.array([np.mean(Dell_V[binedges[i] : binedges[i + 1] + 1]) for i in range(len(binedges) - 1)])
    
    return DV


class Vansyngel_Model:
    # Dust model class for inference and evaluation
    
    def __init__(self):
        # Storage for MCMC samples from fitting the model
        self.samples = None
        self.Nsamples = 0
        
        self.param_names = ['p0', 'alphaM', 'fM']
        
        # plot labels
        self.param_labels = [r'$p_0$', r'$\alpha_M$', r'$f_M$']
        
        self.priors = {'p0':None, 'alphaM':None, 'fM':None}
  
    def log_prior(self, **params):
        """Uniform log prior PDF p(params|H)

        Args:
            params: dictionary of model parameters
        """
        return np.sum(st.uniform.logpdf([params['p0'], params['alphaM'], params['fM']],\
                                       loc=[0, -5, 0], scale=[0.5, 4, 2]))

    def draw_samples_from_prior(self, Ns):
        """Draw samples from the prior PDF as a list of dictionaries

        Args:
            Ns: Number of samples to draw
        """
        samples = []
        for i in range(Ns):
            self.priors['p0'] = st.uniform.rvs(loc=0, scale=0.5)
            self.priors['alphaM'] = st.uniform.rvs(loc=-5, scale=4)
            self.priors['fM'] = st.uniform.rvs(loc=0, scale=2)
            samples.append(self.priors.copy())

        return samples

    def log_likelihood(self, **params):
        """log likelihood PDF p(y|params,H) using each bin in multipole as a normal
            distribution around the BICEP/Keck (2015) best-fit dust model values

        Args:
            params: dictionary of model parameters
        """
        Dell_V = np.mean(self.generate_replica_dataset(7, **params), axis = 0)

        return np.sum(st.norm.logpdf(Dell_V, loc = BKexp, scale = BKstd))
        
    def generate_replica_dataset(self, Ns, **params):
        """Generate a dataset from the sampling distribution
        
        Args:
            Ns: Number of samples to generate
            params: dictionary of model parameters
        """
        DV = []
        for i in range(Ns):
            DV.append(make_dustsim(**params))
            
        return np.array(DV)
        
    def log_posterior(self, parameterlist = None, **params):
        """log of the unnormalized posterior PDF p(params|y,H)

        Args:
            parameterlist: for compatibility with emcee
            params: dictionary of model parameters
        """
        if parameterlist is not None:
            pdict = {k:parameterlist[i] for i,k in enumerate(self.param_names)}
            return self.log_posterior(**pdict)
        lnp = self.log_prior(**params)
        if lnp != -np.inf:
            lnp += self.log_likelihood(**params)

        return lnp

    def draw_samples_from_posterior(self, starting_params, nsteps, nwalkers=6):
        """Use emcee to draw samples from P(params|y,H)

        Args:
            starting_params: initial guess
            nsteps: Number of MCMC steps to run for each walker
            nwalkers: Number of walkers (use >= twice # of parameters)
        """
      
        npars = len(starting_params)   

        # Generate an ensemble of walkers within +/-1% of the guess:
        theta_0 = np.array([starting_params*(1.0 + 0.01 * np.random.randn(npars)) for j in range(nwalkers)])
        # Note that the initial parameter array theta_0 should have dimensions nwalkers × npars
        
        # Use all cores available
        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(nwalkers, npars, self.log_posterior, pool = pool) 
            # Evolve the ensemble:
            self.sampler.run_mcmc(theta_0, nsteps, progress=True)
        
        # Plot the raw samples
        plt.rcParams['figure.figsize'] = (12.0, 4.0*npars)
        fig, ax = plt.subplots(npars, 1);
        cr.plot_traces(self.sampler.chain[:min(8,nwalkers),:,:], ax, labels = self.param_labels);
        plt.savefig("trace_plots.png")

    def check_chains(self, burn, maxlag):
        '''Ignore burn-in samples at the start of each chain and compute convergence criteria and
            effective number of samples.

        Args:
            burn: number of sample to ignore at the front of each chain
            maxlag: a guess at the maximum lag in chain autocorrelation since doing the calculation to 
                arbitrarily long lags becomes very expensive
        '''
        nwalk, nsteps, npars = self.sampler.chain.shape
        if burn < 1 or burn >= nsteps:
            return
        tmp_samples = [self.sampler.chain[i,burn:,:] for i in range(nwalk)]
        print('R =', cr.GelmanRubinR(tmp_samples))
        print('neff =', cr.effective_samples(tmp_samples, maxlag = maxlag))
        print('NB: Since walkers are not independent, these will be optimistic!')

    def remove_burnin(self, burn):
        '''Remove burn-in samples at the start of each chain and concatenate.
            Plot, and store the result in self.samples

        Args:
            burn: number of sample to remove at the front of each chain
        '''
        nwalk, nsteps, npars = self.sampler.chain.shape
        if burn < 1 or burn >= nsteps:
            return
        self.samples = self.sampler.chain[:,burn:,:].reshape(nwalk*(nsteps-burn), npars)
        self.Nsamples = self.samples.shape[0]
        plt.rcParams['figure.figsize'] = (12.0, 4.0 * npars)
        fig, ax = plt.subplots(npars, 1);
        cr.plot_traces(self.samples, ax, labels = self.param_labels);
        
    def posterior_mean(self):
        '''Compute the posterior mean of each parameter from MCMC samples
        '''
        m = np.mean(self.samples, axis = 0)
        return {k:m[i] for i,k in enumerate(self.param_names)}
    

if __name__ == '__main__':

    nside = 256
    d2r = np.pi / 180.
    rot = hp.Rotator(coord=['G','C'])
    lmax = 336
    ell = np.arange(lmax+1)
    
    # Load Planck 353 GHz intensity map
    dust_map = hp.read_map('/home/groups/clkuo/planck_maps_for_george/pr3/353/HFI_SkyMap_353_2048_R3.01_full.fits',0)
    dust_map = hp.ud_grade(dust_map, nside)
    dust_map *= 1e6 # These maps are given in K_{CMB}
    dust_map = rot.rotate_map_pixel(dust_map)
    
    # BICEP/Keck mask
    mask = hp.read_map('/home/groups/clkuo/planck_maps_for_george/masks/bk18_mask_smallfield_cel_n0512.fits')
    mask = hp.ud_grade(mask, nside)
    mask = np.nan_to_num(mask)
    mask[mask < 0] = 0
    
    fsky = np.sqrt(np.mean(mask**2))
    
    npix = hp.nside2npix(nside)
    r = hp.pix2vec(nside,np.arange(npix))
    
    # North vector n = (r x z) x r (z = [0,0,1]) 
    n = r * r[2] * (-1.)
    n[2] += 1.
    n = n / np.sqrt(np.einsum('sp,sp->p', n, n))
    
    # East vector e = -r x n
    e = np.cross(r, n, axisa=0, axisb=0).T * (-1.)
    e = e / np.sqrt(np.einsum('sp,sp->p', e, e))
    
    # BICEP/Keck 2015 mean and standard deviation values for the best-fit dust model in BB over multipole bins
    BKexp = np.array([0.1674, 0.1272, 0.1016, 0.0870, 0.0773, 0.0706, 0.0652, 0.0605, 0.0566])
    BKstd = np.array([0.1796, 0.0850, 0.0707, 0.0845, 0.1108, 0.1442, 0.1822, 0.2480, 0.3606])
    binedges = np.array([20, 55, 90, 125, 160, 195, 230, 265, 300, 335])
    bincenters = np.array([(binedges[i]+binedges[i+1])/2 for i in range(9)])

    Model = Vansyngel_Model()

    # Carry out the parameter inference and display the Markov chains
    start = time.time()
    Model.draw_samples_from_posterior(starting_params=[0.26, -2.5, 0.9], nsteps=1000, nwalkers=6)
    print("emcee took {0:.1f} seconds".format(time.time() - start))
    
    # Compute the convergence criteria and effective number of samples
    Model.check_chains(burn=100, maxlag=500)

    # Remove burn-in for good and plot the concatenation of what's left
    Model.remove_burnin(burn=100)

    # Check the covariances of the parameters
    plotGTC(Model.samples, paramNames=Model.param_labels, figureSize=6,
            customLabelFont={'size':12}, customTickFont={'size':12}, customLegendFont={'size':16})

    print("Posterior mean parameters = ", Model.posterior_mean())

    # Save session
    dill.dump_session('fitted_model.db')
