# coding=utf-8

from basenmf import BaseNMF, NMFResult
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.neighbors import kneighbors_graph

class NSpecClus(BaseNMF):
    """
    An implementation of Non-negative Spectral Clustering, as presented in

    Ding, C., Tao Li, and M.I. Jordan. “Nonnegative Matrix Factorization for Combinatorial Optimization:
    Spectral Clustering, Graph Matching, and Clique Finding.” In Eighth IEEE International Conference on
    Data Mining, 2008. ICDM ’08, 183–92, 2008. doi:10.1109/ICDM.2008.130.

    The class initializer DOES NOT take an affinity matrix, but rather takes the original data matrix, computes
    the affinity matrix via a Gaussian kernel, and performs clustering on that front.

    """

    def __init__(self, X, k, **kwargs):
        """

        :param X: The data matrix, **must** be a data matrix of shape (n_samples, n_features)
        :param k: number of clusters
        :return:
        """
        BaseNMF.__init__(self, X, k, **kwargs)

        _affinity = kwargs.get("affinity")
        if _affinity:
            if _affinity == "nn":
                self._affinity = "nn"
            elif _affinity == "gaussian":
                # will compute the Gaussian kernel of euclidean distance
                self._affinity = "gaussian"
            else:
                raise Exception("Unrecognized affinity parameter %s given" % _affinity)
        else:
            self._affinity = "nn"

        # the gamma parameter for Gaussian kernel, default 1.
        self._gamma = kwargs.get("gamma") or 1.


    def predict(self):

        if self._affinity == "gaussian":
            # first convert the data matrix to pairwise distances
            self.dist = squareform(pdist(self.X))
            # then compute the affinity matrix using Gaussian kernel of the Euclidean distance
            self.V = np.exp(-self._gamma * self.dist ** 2)

        else:
            # calculate the K-NN graph of the matrix
            self.V = kneighbors_graph(self.X)

        self.V = np.matrix(self.V)
        m, n = self.V.shape
        dd = np.array(self.V.sum(1))[:,0]
        D = np.matrix(np.diag(dd))
        H = np.matrix(np.random.rand(m, self.k))

        convgraph = np.zeros(self.maxiter / 10)
        distold = 0.

        for i in range(self.maxiter):

            # multiplicative update step, Euclidean error reducing
            alpha = H.T * self.V * H

            d1 = np.sqrt(np.divide(self.V*H, D*H*alpha))
            H = np.multiply(H, d1)

            # every 10 iterations, check convergence
            if i % 10 == 0:
                dist = alpha.trace()
                convgraph[i/10] = dist

                diff = dist - distold
                print "diff is %s" % diff

                distold = dist

        return NMFResult((H,), convgraph, distold)
