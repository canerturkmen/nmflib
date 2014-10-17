from basenmf import BaseNMF, NMFResult
from metrics import frobenius, kldivergence
import sys
import numpy as np

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

        for i in range(self.maxiter):
            if self.objective == "eu":
                # multiplicative update steps, Euclidean error reducing
                H = H * ((np.dot(W.T,V)/np.dot(np.dot(W.T, W), H )))
                W = W * ((np.dot(V, H.T)/np.dot(np.dot(W, H), H.T )))

            elif self.objective == "kl":
                # multiplicative update steps for KL Divergence
                H = H * (np.dot(W.T, (V / np.dot(W, H)))) /np.tile(np.sum(W,0), (n,1)).T
                W = W * (np.dot((V / np.dot(W, H)), H.T)) /np.tile(np.sum(H,1), (m,1))

            # every 10 iterations, check convergence
            if i % 10 == 0:
                if self.objective == "eu":
                    dist = frobenius(V, np.dot(W,H))

                elif self.objective == "kl":
                    dist = kldivergence(V, np.dot(W,H))

                convgraph[i/10] = dist
                # print dist
                if distold - dist < self.stopconv:
                    print "Converged"
                    break
                distold = dist

        return NMFResult((W, H), convgraph, dist)
