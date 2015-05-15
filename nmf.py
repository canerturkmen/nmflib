from basenmf import BaseNMF, NMFResult
from .utils import frobenius, kldivergence, normalize_factor_matrices
import sys
import numpy as np
#%%
class NMF(BaseNMF):
    """
    Implementation of basic NMF (Lee and Seung) algorithm with Euclidean and KL-divergence objective functions
    """

    objective = "eu" # objective function to optimize on

    def __init__(self, X, k, **kwargs):
        BaseNMF.__init__(self, X, k, **kwargs)

        if kwargs.get("metric"):
            self.objective = kwargs.get("metric")

    def predict(self):
        """
        Euclidean distance reducing update rules for NMF, presented in Lee and Seung (2001)
        """

        m, n = self.X.shape
        V = self.X
        distold = sys.maxint #very large number

        W = np.random.rand(m, self.k)
        H = np.random.rand(self.k, n)
        convgraph = np.zeros(self.maxiter / 10)

        eps = 1e-9 # small number for stability

        for i in range(self.maxiter):
            # multiplicative update steps, Euclidean error reducing
            H = H * ( W.T.dot(V) + eps / W.T.dot(W).dot(H) + eps  )
            W = W * ( V.dot(H.T) + eps / W.dot(H.dot(H.T)) + eps  )

            # every 10 iterations, check convergence
            if i % 10 == 0:
                dist = frobenius(V, W.dot(H))
                convgraph[i/10] = dist

        W, H = normalize_factor_matrices(W,H)

        return NMFResult((W, H), convgraph, dist)
