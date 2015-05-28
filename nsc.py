# coding=utf-8

from basenmf import BaseNMF, NMFResult
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.sparse import issparse, csr_matrix, diags
from scipy.sparse.linalg import eigs
from .utils import trace as sptrace, norm as sparse_norm
# import pdb

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

        _affinity = kwargs.get("affinity") or "nn"
        _gamma = kwargs.get("gamma") or 1.
        _nn = kwargs.get("nn") or 10

        self._embedding = kwargs.get("embedding") or True
        # Derive the affinity matrix

        if _affinity == "hybrid":
            # the K-NN matrix weighted by Gaussian affinity
            knng = kneighbors_graph(self.X, n_neighbors=_nn)
            dist_matrix = squareform(pdist(X, "sqeuclidean"))
            affinity = csr_matrix(np.exp(-_gamma * dist_matrix))
            self.A = affinity.multiply(knng)

        elif _affinity == "nn":
            # The affinity matrix is a sparse nearest neighbors graph {0,1}^(n x n)
            self.A = kneighbors_graph(self.X, n_neighbors=_nn)

        elif _affinity == "gaussian":
            dist_matrix = squareform(pdist(X, "sqeuclidean"))
            self.A = csr_matrix(np.exp(-_gamma * dist_matrix))

        elif _affinity == "linear":
            # Derive the pairwise inner product kernel matrix
            self.A = self.X * self.X.T

        else:
            raise ValueError("Unrecognized kernel given")

    def predict(self):

        m, n = self.A.shape # m observations

        convgraph = np.zeros(self.maxiter / 25)
        prevdist = 0.
        converged = False

        eps = 1e-6

        dd = np.array(self.A.sum(1))[:,0]
        D = diags(dd,0, format="csr")

        m, n = self.A.shape


        # random initialization, will initialize with K-means if told to
        H = csr_matrix(np.random.rand(m, self.k))

        EPS = csr_matrix(np.ones(H.shape) * eps)

        if self._embedding:
            # Apply eigenspace embedding K-means for initialization (Ng Weiss Jordan)

            Dz = diags(1 / (np.sqrt(dd) + eps), 0, format="csr")
            DAD = Dz.dot(self.A).dot(Dz)

            V = eigs(DAD, self.k)[1].real
            km_data = V / (np.linalg.norm(V, 2, axis=1).T * np.ones((self.k,1))).T

            km_predict = KMeans(n_clusters=self.k).fit_predict(km_data)

            indices = km_predict
            indptr = range(len(indices)+1)
            data = np.ones(len(indices))
            H = csr_matrix((data, indices, indptr))

        # Run separately for sparse and dense versions

        for i in range(self.maxiter):

            AH = self.A.dot(H)
            alpha = H.T.dot(AH)

            M1 = AH + EPS
            M2 = D.dot(H).dot(alpha) + EPS

            np.reciprocal(M2.data, out=M2.data)
            d1 = M1.multiply(M2).sqrt()

            H = H.multiply(d1)

            if i % 25 == 0:
                dist = sptrace(alpha)
                convgraph[i/25] = dist

                diff = dist / prevdist - 1
                prevdist = dist

        return NMFResult((H.toarray(),), convgraph, pdist)
