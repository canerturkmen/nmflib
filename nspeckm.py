# coding=utf-8
"""
A Hybrid of Nonnegative Spectral Clustering with KMeans.

In our original implementation of Non-Negative Spectral Clustering,
we had a feature correspond to each of the clusters, and we chose the feature that
was maximum as the instance's cluster.

In this implementation, we will use Nonnegative Spectral Clustering with K-NN graph,
and derive ~K*1.2 features. Then we will use a naive clustering algorithm (e.g. KMeans) to
generate the final clusters.
"""

from basenmf import BaseNMF, NMFResult
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, diags, issparse
#%%
class NSpecSparseKM(BaseNMF):
    """
    An implementation of Non-negative Spectral Clustering, with K-means used as the final
    clustering algorithm on ~K*1.2 features..
    """

    def __init__(self, X, k, **kwargs):
        """
        :param X: The data matrix, **must** be a data matrix of shape (n_samples, n_features)
        :param k: number of clusters
        :return:
        """
        self._orig_k = k
        # we increment the K slightly (empirical observation - S Sanghavi)
        BaseNMF.__init__(self, X, int(k*1.2), **kwargs)
        # the gamma parameter for Gaussian kernel, default .005
        self._gamma = kwargs.get("gamma") or .005

    def _check_nans(self, X, name):
        if issparse(X):
            X = X.todense()
        if np.any(np.isnan(X)):
            raise Exception("First NaN encountered at %s" % name)


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

            VH = self.V*H # 486

            alpha = H.T * VH # 272

            d1 = csr_matrix(np.sqrt(np.divide(VH + EPS, D*H*alpha + EPS)))
            H = H.multiply(d1) #20ms

            if i % 10 == 0:

                dist = alpha.todense().trace()
                #print dist
                convgraph[i/10] = dist

                diff = dist - distold
                #print "diff is %s" % diff

                distold = dist

        # H is a matrix

        km = KMeans(n_clusters=self._orig_k, n_jobs=2)
        centers = km.fit_predict(H)

        return NMFResult(centers, convgraph, distold)

