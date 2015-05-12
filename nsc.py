# coding=utf-8

from basenmf import BaseNMF, NMFResult
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import issparse, csr_matrix, diags
from .utils import trace as sparse_trace

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
        self._gamma = kwargs.get("gamma") or .005
        # Derive the affinity matrix

        if _affinity == "nn" or _affinity is None:
            # The affinity matrix is a sparse nearest neighbors graph {0,1}^(n x n)
            self.A = kneighbors_graph(self.X, n_neighbors=20)
        elif _affinity == "gaussian":
            dist_matrix = squareform(pdist(self.X))
            self.A = np.exp(-self._gamma * dist_matrix ** 2)
        elif _affinity == "linear":
            # Derive the pairwise linear kernel matrix
            self.A = self.X * self.X.T
        else:
            raise ValueError("Unrecognized kernel given")

    def predict(self):

        m, n = self.A.shape # m observations

        convgraph = np.zeros(self.maxiter / 10)
        pdist = 0.
        converged = False

        dd = np.array(self.A.sum(1))[:,0]
        H = np.matrix(np.random.rand(m, self.k))

        # Run separately for sparse and dense versions
        if issparse(self.A):
            # We are working on a nearest neighbors graph

            D = diags(dd,0, format="csr")
            H = csr_matrix(H)

            EPS = csr_matrix(np.ones(H.shape)*np.finfo(float).eps) # matrix of epsilons

            for i in range(self.maxiter):

                AH = self.A*H # 486

                alpha = H.T * AH # 272

                d1 = csr_matrix(np.sqrt(np.divide(AH + EPS, D*H*alpha + EPS)))
                H = H.multiply(d1) #20ms

                if i % 10 == 0:
                    dist = sparse_trace(alpha)
                    convgraph[i/10] = dist
                    pdist = dist
        else:
            self.A = np.matrix(self.A)

            D = np.matrix(np.diag(dd))

            for i in range(self.maxiter):

                # multiplicative update step, Euclidean error reducing
                alpha = H.T * self.A * H

                d1 = np.sqrt(np.divide(self.A*H, D*H*alpha))
                H = np.multiply(H, d1)

                # every 10 iterations, check convergence
                if i % 10 == 0:
                    dist = alpha.trace()
                    convgraph[i/10] = dist
                    pdist = dist

        return NMFResult((H,), convgraph, pdist)
