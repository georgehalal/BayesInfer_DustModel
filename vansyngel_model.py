# -*- coding: utf-8 -*-
"""
VansyngelModel

Implementation of the Vansyngel model.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"
__all__ = ["VansyngelModel"]


from dataclasses import dataclass

import healpy as hp
import numpy as np
import pysm


@dataclass
class VansyngelModel:
    """Vansyngel Dust Model Implementation
    """

    def __init__(self, dust_map: np.ndarray, mask: np.ndarray, p_0: float,
                 alpha_m: float, f_m: float) -> None:
        """Define necessary variables based on the input arguments

        Parameters:
            dust_map (np.ndarray): dust intensity, e.g. Planck 353 GHz
            mask (np.ndarray): BICEP/Keck mask
            p_0 (float): effective polarization fraction of dust
                emission
            alpha_m (float): power-law index of correlated-Gaussian
                field power spectrum for Bt
            f_m (float): relative strength of the random component of
                the field
        """
        self.p_0 = p_0
        self.alpha_m = alpha_m
        self.f_m = f_m
        self.dust_map = dust_map
        self.nside = hp.get_nside(dust_map)

        if hp.get_nside(mask) != self.nside:
            mask = hp.ud_grade(mask, self.nside)
        self.mask = mask

        self.fsky = np.sqrt(np.mean(self.mask ** 2))

        self.npix = hp.nside2npix(nside)
        self.r = hp.pix2vec(nside, np.arange(self.npix))

        # North vector n = (r x z) x r (z = [0,0,1]) 
        n = self.r * self.r[2] * (-1.)
        n[2] += 1.
        self.n = n / np.sqrt(np.einsum("sp,sp->p", n, n))

        # East vector e = -r x n
        e = np.cross(self.r, n, axisa=0, axisb=0).T * (-1.)
        self.e = e / np.sqrt(np.einsum("sp,sp->p", e, e))

        self.lmax = 3 * self.nside - 1
        self.ell = np.arange(self.lmax + 1)
        # BICEP/Keck binning
        self.binedges = np.array(
            [20, 55, 90, 125, 160, 195, 230, 265, 300, 335])
        
        # Galactic longitude and latitude defining B0 orientation
        self.l0 = np.radians(70)
        self.b0 = np.radians(24)

        # number of shells approximating integral along line-of-sight
        self.N = 7

        return None

    def do_projection(self, B: np.ndarray, 
                      Bdims: str = "sp") -> tuple[np.ndarray, np.ndarray]:
        """Project the magnetic field onto a sphere

        Parameters:
            B (np.ndarray): magnetic field
            Bdims (str): dimensions of magnetic field

        Returns:
            (tuple[np.ndarray, np.ndarray]): GMF phi and (cos(gamma))^2
        """
        B /= np.sqrt(np.einsum("%s,%s->p"%(Bdims, Bdims), B, B))
        Bperp = B - self.r * np.einsum("%s,sp->p"%Bdims, B, self.r)
        phi = np.pi - np.sign(np.einsum("sp,sp->p", Bperp, self.e)) * np.arccos(np.einsum("sp,sp->p",
            Bperp, self.n) / np.sqrt(np.einsum("sp,sp->p", Bperp, Bperp)))
        cos2gamma = 1. - np.abs(np.einsum("%s,sp->p"%Bdims, B, self.r))**2
    
        return phi, cos2gamma
    
    def simulate_GMF(self) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the Galactic magnetic field as a sum of mean and
        turbulent components (B = B0 + f_m Bt)

        Returns:
            (tuple[np.ndarray, np.ndarray]): GMF phi and (cos(gamma))^2
        """
        B0 = np.array([np.cos(self.l0) * np.cos(b0),
                       np.sin(self.l0) * np.cos(self.b0),
                       np.sin(self.b0)])
        
        phi = np.zeros((self.N, self.npix))
        cos2gamma = np.zeros((self.N, self.npix))
        
        Cell = self.ell ** self.alpha_m
        Cell[:2] = 0.
        
        for i in range(self.N):
            while True:
                Bt = np.array([hp.synfast(Cell, self.nside, verbose=False),
                               hp.synfast(Cell, self.nside, verbose=False),
                               hp.synfast(Cell, self.nside, verbose=False)])
                Bt /= np.sqrt(np.einsum("sp,sp->p", Bt, Bt))
                B = B0[:, np.newaxis] + self.f_m * Bt
                phi[i], cos2gamma[i] = do_projection(B)
                if (np.sum(np.isnan(phi[i]))
                        + np.sum(np.isnan(cos2gamma[i]))) == 0:
                    break
                    
        return cos2gamma, phi
    
    def make_dustsim(self) -> np.ndarray:
        """Make a realization of dust polarization power spectrum

        Returns:
            (np.ndarray): a realization of dust polarization power
                spectrum
        """
        dust_sim = np.zeros((3, self.npix))
        
        cos2gamma, phi = simulate_GMF()
        
        S = self.dust_map / np.sum(
            1. - (cos2gamma - 2. / 3.) * self.p_0, axis=0)
        
        # Use Planck 353 GHz as dust intensity
        dust_sim[0] = self.dust_map
        dust_sim[1] = S * self.p_0 * np.sum(
            np.cos(2. * phi) * cos2gamma, axis=0)
        dust_sim[2] = S * self.p_0 * np.sum(
            np.sin(2. * phi) * cos2gamma, axis=0)
        
        # Use PySM to scale to 220 GHz
        sky = pysm.Sky(nside=self.nside, preset_strings=["d0"])
        sky.components[0].I_ref = dust_sim[0] * pysm.units.uK_CMB
        sky.components[0].Q_ref = dust_sim[1] * pysm.units.uK_CMB
        sky.components[0].U_ref = dust_sim[2] * pysm.units.uK_CMB
        sky.components[0].freq_ref_P = sky.components[0].freq_ref_I
        model_freq_maps = sky.get_emission(220 * pysm.units.GHz)
        
        Dell_V = (self.ell * (self.ell+1)
                  * hp.anafast(self.mask * model_freq_maps, lmax=self.lmax)[2]
                  / self.fsky / 2 / np.pi)
        
        # Use BICEP/Keck binning
        return np.array([np.mean(
            Dell_V[self.binedges[i]:self.binedges[i + 1] + 1]) for i in range(
                len(self.binedges) - 1)])
