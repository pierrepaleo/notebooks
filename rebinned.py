#!/usr/bin/env python
# coding: utf-8

import numpy as np
from spire.utils import *
from spire.operators.image import *
from spire.tomography.tomography import AstraToolbox
from spire.algorithms.chambollepock import chambolle_pock_tv, chambolle_pock_tv_wavelets
from spire.operators.misc import power_method
from time import time
from pypwt import Wavelets





#
# ATTENTION chambolle_pock_tv_wavelets   utilise dta=False et normalize=True,  et on veut l'inverse !
#   pour ce test, ça fonctionne assez bien cependant
#

def bin12(x):
    """
    Bin along axis=-1 only
    """
    return x.reshape(x.shape[0], x.shape[1]//2, 2).mean(axis=-1)


def zoom2(img):
    return np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(img)), (img.shape[0]*2, img.shape[1]*2)))*4.


class ReconstructionMethod(object):

    algorithms = {
        "TV": chambolle_pock_tv,
        "TV+W": chambolle_pock_tv_wavelets,
        "TV/rebinned": chambolle_pock_tv,
        "TV+W/rebinned": chambolle_pock_tv_wavelets,
    }

    def __init__(self, type, sinogram, groundtruth):
        """

        type: str
            can be "TV", "TV+W", "TV/rebinned", "TV+W/rebinned"
        """

        if type not in self.algorithms.keys():
            raise ValueError("type must be in %s" % str(self.types))
        self.type = type
        self.rebinned = "rebinned" in type # TODO: cleaner way
        self.groundtruth = np.copy(groundtruth)
        self.sinogram = np.copy(sinogram) / sinogram.shape[0]*np.pi/2  # Rescaling

        self.tomo = AstraToolbox(self.sinogram.shape[1], self.sinogram.shape[0], cudafbp=True)
        self.tomo.set_filter("hamming") # preconditioner
        self.P = lambda x : self.tomo.proj(x)/self.tomo.n_a*np.pi/2
        self.PT = lambda x : self.tomo.fbp(x) # no rescaling since using astra FBP

        dummy_slice = np.zeros((sino.shape[1], sino.shape[1]))
        self.use_wavelets = ("W" in type) # TODO: cleaner way
        if self.use_wavelets:
            self.W = Wavelets(dummy_slice, "haar", 99) # should be orthogonal (no cycle spin)

        if self.rebinned:
            self.sinogram_binned = bin12(self.sinogram)
            self.tomo_rebinned = AstraToolbox(self.sinogram_binned.shape[1], self.sinogram_binned.shape[0], cudafbp=True)
            self.tomo_rebinned.set_filter("hamming") # preconditioner
            self.P2 = lambda x : self.tomo_rebinned.proj(x)/self.tomo_rebinned.n_a*np.pi/2
            self.P2T = lambda x : self.tomo_rebinned.fbp(x) # no rescaling since using astra FBP
            if self.use_wavelets:
                self.W2 = Wavelets(np.zeros((dummy_slice.shape[0]//2, dummy_slice.shape[1]//2)), "haar", 99)

        self.results = []
        self.images = []
        # Pre-compute primal/dual steps for Chambolle-pock
        self.opnorm = sqrt(8 + power_method(self.P, self.PT, self.sinogram, 20)**2)*1.2
        if self.rebinned:
            self.opnorm_binned = sqrt(8 + power_method(self.P2, self.P2T, self.sinogram_binned, 20)**2)*1.2


    def benchmark(self, n_it, beta_tv, n_it2=None, beta_wavelets=None):
        """
        Benchmark the current algorithm.

        Parameters
        -----------

        n_it: number of iterations for the main algorithm
        beta_tv: regularization parameter for TV
        n_it2 (optionnal): number of iterations for the binned version (if relevant)
        beta_wavelets (optionnal): regularization parameter for wavelets (if relevant)
        """

        # TODO if self.verbose
        print("Running %s" % self.type)

        algo = self.algorithms[self.type]
        # TODO: it would be better to rescale the regularization parameter(s) when performing the first binned reconstruction
        if not(self.rebinned):
            x0 = None
            elb = None
        else:
            t0 = time()
            if self.use_wavelets:
                x0 = algo(self.sinogram_binned, self.P2, self.P2T, self.W2, beta_tv, beta_wavelets, L=self.opnorm_binned, n_it=n_it2, return_all=False)
            else:
                x0 = algo(self.sinogram_binned, self.P2, self.P2T, beta_tv, L=self.opnorm_binned, n_it=n_it2, return_all=False)
            elb = (time() - t0)*1.0
            x0 = zoom2(x0/2.) # TODO: binning only in one direction makes this scaling necessary => have to adapt the beta !

        t0 = time()
        if self.use_wavelets:
            res = algo(self.sinogram, self.P, self.PT, self.W, beta_tv, beta_wavelets, L=self.opnorm, n_it=n_it, return_all=False, x0=x0)
        else:
            res = algo(self.sinogram, self.P, self.PT, beta_tv, L=self.opnorm, n_it=n_it, return_all=False, x0=x0)
        el = (time() - t0)*1.0

        mse = norm2sq(res - self.groundtruth)/res.size

        infos = {
            "iterations_binned": n_it2,
            "iterations": n_it,
            "beta_tv": beta_tv,
            "beta_wavelets": beta_wavelets,
            "time_binned": elb,
            "time": el,
            "mse": mse,
        }
        self.results.append(infos)
        self.images.append(res) # TODO: avoid if many benchmarks !




if __name__ == "__main__":

    ph = np.load("data/Brain512.npz")["data"]
    tomo = AstraToolbox(ph.shape[1], 40)
    sino = tomo.proj(ph)
    sino = sino + np.random.randn(*sino.shape)*sino.max()*1.0/100

    R_tv = ReconstructionMethod("TV", sino, ph)
    R_tv.benchmark(500, 0.9)

    R_tv_b = ReconstructionMethod("TV/rebinned", sino, ph)
    R_tv_b.benchmark(100, 0.9, n_it2=500)

    print(R_tv.results)
    print(R_tv_b.results)

    ims([R_tv.images[0], R_tv_b.images[0]], cmap="gray")





"""

Method          num_it          Time (s)        MSE     Comments
TV
TV/rebinned
TV+W/
TV+W/rebinned





"""
