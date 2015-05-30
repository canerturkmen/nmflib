from basenmf import BaseNMF, NMFResult
from .utils import frobenius, kldivergence, normalize_factor_matrices
import numpy as np
#%%
class NMF(BaseNMF):
    """
    Implementation of basic NMF (Lee and Seung) algorithm with Euclidean and KL-divergence objective functions
    """

    def predict(self):
        """
        Euclidean distance reducing update rules for NMF, presented in Lee and Seung (2001)
        """

        m, n = self.X.shape
        V = self.X
        pdist = 1e9 #very large number

        W = np.random.rand(m, self.k)
        H = np.random.rand(self.k, n)
        convgraph = np.zeros(self.maxiter / 10)
        converged = False

        eps = 1e-7 # small number for stability

        for i in range(self.maxiter):
            # multiplicative update steps, Euclidean error reducing
            H = H * (( W.T.dot(V) + eps) / (W.T.dot(W).dot(H) + eps))
            H = (H.T / np.linalg.norm(H, 2, 1)).T

            W = W * ( (V.dot(H.T) + eps) / (W.dot(H.dot(H.T)) + eps) )
            W /= np.linalg.norm(W, 2, 0)
            # normalize columns of H and W

            # every 10 iterations, check convergence
            if i % 100 == 0:
                dist = frobenius(V, W.dot(H))
                convgraph[i/10] = dist

                # print dist

                if pdist - dist < self.stopconv and pdist - dist > self.stopconv:
                    converged = True
                    break

                pdist = dist

        W, H = normalize_factor_matrices(W,H)

        return NMFResult((W, H), convgraph, dist)
