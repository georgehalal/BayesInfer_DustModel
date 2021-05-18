# Bayesian Inference of Vansyngel et al. Dust Model Parameters in the BICEP/Keck Footprint

## Introduction
To model the polarized Galactic emission, we discretize the integral equations for the emission of an optically thin medium [[Planck 2015](https://www.aanda.org/articles/aa/pdf/2015/04/aa24086-14.pdf)] as is done in [Vansyngel et al. 2016](https://arxiv.org/pdf/1611.02577.pdf). We assume the same source function, <img src="https://render.githubusercontent.com/render/math?math=S">, in each layer and factor out the frequency dependence into pre-factors, <img src="https://render.githubusercontent.com/render/math?math=A_\nu">, resulting in

<img src="https://render.githubusercontent.com/render/math?math=I(\nu) = A_\nu S \sum_i^N \left[1 - p_0 \left(\cos^2{\gamma_i}-\frac{2}{3}\right)\right]">

<img src="https://render.githubusercontent.com/render/math?math=Q(\nu) = A_\nu S \sum_i^N p_0 \cos{(2\phi_i)}\cos^2{\gamma_i}">

<img src="https://render.githubusercontent.com/render/math?math=U(\nu) = A_\nu S \sum_i^N p_0 \sin{(2\phi_i)}\cos^2{\gamma_i}">

where <img src="https://render.githubusercontent.com/render/math?math=p_0"> is a parameter describing the dust polarization properties and <img src="https://render.githubusercontent.com/render/math?math=\gamma"> and <img src="https://render.githubusercontent.com/render/math?math=\phi"> describe the orientation of the Galactic magnetic field with respect of the plane of the sky. We use <img src="https://render.githubusercontent.com/render/math?math=N=7"> as fit by Planck data.

We model the Galactic magnetic field, following [Planck 2016](https://arxiv.org/abs/1604.01029), as <img src="https://render.githubusercontent.com/render/math?math=B = B_0"> + <img src="https://render.githubusercontent.com/render/math?math=B_t">, where <img src="https://render.githubusercontent.com/render/math?math=B_0"> is the mean component and <img src="https://render.githubusercontent.com/render/math?math=B_t = |B_0|f_M\hat{B_t}"> is the turbulent component. Far away from the Galactic disk, we can assume <img src="https://render.githubusercontent.com/render/math?math=B_0"> to have a fixed orientation. For that, we use the Galactic longitude and latitude of the mean component of the Galactic magnetic field fit using Planck data. The turbulent field orientation is simulated by drawing each component of the 3D vector in each direction on the sky from a correlated Gaussian, with variance <img src="https://render.githubusercontent.com/render/math?math=C_\ell \sim \ell^{\alpha_M}">. We project this magnetic field onto the two-dimensional sky sphere to get <img src="https://render.githubusercontent.com/render/math?math=\gamma"> and <img src="https://render.githubusercontent.com/render/math?math=\phi">.  

We use MCMC methods to perform Bayesian inference on <img src="https://render.githubusercontent.com/render/math?math=p_0">, <img src="https://render.githubusercontent.com/render/math?math=f_M">, and <img src="https://render.githubusercontent.com/render/math?math=\alpha_M"> in the BICEP/Keck footprint by fitting to the best-fit dust model from [BICEP/Keck X](https://arxiv.org/abs/1810.05216). Using the Planck 353 GHz intensity map for <img src="https://render.githubusercontent.com/render/math?math=I(\nu)">, we can calculate <img src="https://render.githubusercontent.com/render/math?math=S"> and use it for calculating <img src="https://render.githubusercontent.com/render/math?math=Q(\nu)"> and <img src="https://render.githubusercontent.com/render/math?math=U(\nu)">.

## Setup
```bash
git clone https://github.com/georgehalal/BayesInfer_DustModel.git
cd BayesInfer_DustModel/
pip install -r requirements.txt
```
