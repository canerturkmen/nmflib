# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:03:08 2014

@author: Caner
"""

#from metrics import NMF, frobenius
import numpy as np
from basenmf import BaseNMF, NMFResult
from .utils import frobenius

class ProjectiveNMF(BaseNMF):
    """
    Python implementation of ``Projective'' Non-negative Matrix Factorization introduced by Yuan and Oja.
    (2005) in Yuan, Zhijian, and Erkki Oja. “Projective Nonnegative Matrix Factorization for Image Compression
    and Feature Extraction.” In Image Analysis. Lecture Notes in Computer Science 3540.
    Springer Berlin Heidelberg, 2005. http://link.springer.com/chapter/10.1007/11499145_35.
    """

    def predict(self):
        """
        Projective NMF training steps, minimizing the objective function
        `frobenius_norm(V - W*W'*V)` where V is the data matrix
        """

        m, n = self.X.shape
        pdist = 1e9 #very large number

        # convert ndarrays to matrices for cleaner code
        V = np.matrix(self.X)
        W = np.matrix(np.random.rand(m, self.k))

        # VV^T calculated ahead of time
        VV = V * V.T

        # flags and counters for checking convergence
        dist = 0
        converged = False
        convgraph = np.zeros(self.maxiter / 10)

        for i in range(self.maxiter):

            # multiplicative update step, Euclidean error reducing
            num = VV * W
            denom = (W * (W.T * VV * W)) + (VV * W * (W.T * W))
            # W = np.multiply(W, np.divide(num, denom))

            W = np.divide(np.multiply(W, num), denom)

            # W = W .* (XX*W) ./ (W*(W'*XX*W) + XX*W*(W'*W));
            # W = W ./ norm(W);

            # normalize W TODO: check if L2 norm working similar to MATLAB
            W /= np.linalg.norm(W, 2)

            # every 10 iterations, check convergence
            if i % 10 == 0:
                dist = frobenius(V, W*W.T*V)
                convgraph[i/10] = dist

                if pdist - dist < self.stopconv:
                    converged = True
                    break

                pdist = dist

        return NMFResult((np.array(W),), convgraph, dist, converged)
