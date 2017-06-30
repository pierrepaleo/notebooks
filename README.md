# About

This is a collection of jupyter notebooks intended to illustrate some aspects of my PhD thesis about regularized methods in iterative tomographic reconstruction. 
It can also be used to reproduce some figures. 


# Requirements
The notebooks were created on a Jupyter 4.2.1 with a Python 2.7 kernel. The python packages should be recent enough in order to use some features (eg. numpy, scipy.sparse). Having a Nvidia GPU is required to run the tomography-related codes.

You should also install the following:

 *  The [ASTRA Toolbox (>= 1.7)](https://github.com/astra-toolbox/astra-toolbox/)
 *  The [spire package](https://github.com/pierrepaleo/spire)
 *  The [pypwt package](https://github.com/pierrepaleo/pypwt)


# Contents

* Benchmark of some optimization algorithms for tomographic reconstruction with TV regularization (admm_vs_cp). The implemented algorithms are FISTA, ADMM and Chambolle-Pock. 
* Coherence of usual sparsifying transform with respect to the Radon transform (coherence and coherence2)
* Compressibility in the compressed sensing framework (compressibility)
* Puzzling numerical experiments: a reproduction of the numerical experiment of Candes Et Al in 2006 about TV regularization in tomographic reconstruction.
* Using the wavelets transform as a sinogram pre-processing technique to denoise the projections (denoise_projections). This is not the Fourier-Wavelet ring artefact remover.
* On the methods to accelerate "FISTA" (continuation_method). The continuation method and other techniques are compared. 

