# coding=utf-8
"""
Sparse implementation of Nonnegative Spectral Clustering
"""

from basenmf import BaseNMF, NMFResult
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix, diags
#%%
class NSpecSparse(BaseNMF):
    """
    An implementation of Non-negative Spectral Clustering, as presented in

    Ding, C., Tao Li, and M.I. Jordan. “Nonnegative Matrix Factorization for Combinatorial Optimization:
    Spectral Clustering, Graph Matching, and Clique Finding.” In Eighth IEEE International Conference on
    Data Mining, 2008. ICDM ’08, 183–92, 2008. doi:10.1109/ICDM.2008.130.

    This implementation of the NSpecClus, also presented in this module's nmflib.nsc.NSpecClus is for K-Nearest Neighbor
    graphs of records, with scipy.sparse implementation.

    """

    def __init__(self, X, k, **kwargs):
        """
        :param X: The data matrix, **must** be a data matrix of shape (n_samples, n_features)
        :param k: number of clusters
        :return:
        """

        BaseNMF.__init__(self, X, k, **kwargs)
        # the gamma parameter for Gaussian kernel, default .005
        self._gamma = kwargs.get("gamma") or .005


    def predict(self):

        self.V = kneighbors_graph(self.X, n_neighbors=20)

        m, n = self.V.shape

        dd = np.array(self.V.sum(1))[:,0]
        D = diags(dd,0, format="csr")
        H = csr_matrix(np.matrix(np.random.rand(m, self.k)))

        EPS = csr_matrix(np.ones(H.shape)*np.finfo(float).eps) # matrix of epsilons
        
        convgraph = np.zeros(self.maxiter / 10)
        distold = 0.

        for i in range(self.maxiter):

            # multiplicative update step, Euclidean error reducing
            print self.V.shape, H.shape
            VH = self.V*H # 486
            alpha = H.T * VH # 272
            d1 = csr_matrix(np.sqrt(np.divide(VH,(D*H*alpha) + EPS))) #519ms profile
            H = H.multiply(d1) #20ms

            # every 10 iterations, check convergence
            if i % 10 == 0:

                dist = alpha.todense().trace()
                print dist
                convgraph[i/10] = dist
                if dist[0,0] is None:
                    raise Exception("There s a problem with your algorithm !!")
                diff = dist - distold
                print "diff is %s" % diff

                distold = dist

        return NMFResult((H,), convgraph, distold)
