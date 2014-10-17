# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:03:08 2014

@author: Caner
"""

#from metrics import NMF, frobenius
import sys
import numpy as np

class ProjectiveNMF(NMF):
    """
    Python implementation of ``Projective'' Non-negative Matrix Factorization introduced by Yuan and Oja. 
    (2005) in Yuan, Zhijian, and Erkki Oja. “Projective Nonnegative Matrix Factorization for Image Compression 
    and Feature Extraction.” In Image Analysis. Lecture Notes in Computer Science 3540. 
    Springer Berlin Heidelberg, 2005. http://link.springer.com/chapter/10.1007/11499145_35.
    """
    
    def predict_projective(self):
        """
        Projective NMF training steps, minimizing the objective function
        frobenius_norm(V - W*W'*V) where V is the data matrix
        """
        
        m, n = self.X.shape
        distold = sys.maxint #very large number

        # convert ndarrays to matrices for cleaner code        
        V = np.matrix(self.X)
        W = np.matrix(np.random.rand(m, self.k))
        
        # VV^T calculated ahead of time
        Vsq = V * V.T
        
        for i in range(self.MAXITER):
            
            Wsq = W * W.T
            
            # multiplicative update step, Euclidean error reducing   
            num = Vsq * W
            denom = (Wsq* Vsq * W) + (Vsq * Wsq * W)
            W = np.multiply(W, np.divide(num, denom))
            
            # normalize W
            W = np.divide(W, np.linalg.norm(W))            
            
            # every 10 iterations, check convergence
            if i % 10 == 0:
                dist = frobenius(V, Wsq*V)
                print dist, self.STOPCONV, dist-distold
#                if distold - dist < self.STOPCONV:
#                    print "converged"
#                    break
                distold = dist
                
        return Wsq*V

#%% application to Lena


from scipy import misc
import pylab as pl

lena = misc.lena()

lena_hat = ProjectiveNMF(lena, 15).predict_projective()

#%%

pl.imshow(lena, cmap="gray")
pl.figure()
pl.imshow(lena_hat, cmap="gray")
