from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer


class ClusterNMF(BaseNMF):
    """
    An implementation of Cluster-NMF, introduced by Ding et al. in

    Chris Ding, Tao Li, and Michael I. Jordan. Convex and semi-nonnegative matrix factorizations.
    Pattern Analysis and Machine Intelligence, IEEE Transactions on, 32(1):45–55, 2010

    ClusterNMF is theoretically similar to Projective NMF, except for the major difference of being able to
    work with negative data matrices as well.
    """

    def predict(self):
        """
        Run the ClusterNMF prediction algorithm
        """

        # Fresh start, as described in the paper
        KMeans(n_clusters=self.k).fit_predict(X)

        H = np.mat(LabelBinarizer().fit_transform(cl)) # transform to cluster indicator matrix
        D_ = np.mat(np.diag(1 / H.sum(0).astype('float64'))) # D^-1

        # initialize the factorizing matrix
        G = H + .2 * np.ones(H.shape)

        # flags and counters for checking convergence
        dist = 0
        converged = 0
        convgraph = np.zeros(self.maxiter / 10)

        for i in range(self.maxiter):

            # multiplicative update step, Euclidean error reducing
            # num = VV * W
            # denom = (W * (W.T * VV * W)) + (VV * W * (W.T * W))
            # W = np.multiply(W, np.divide(num, denom))

            # normalize W
            W /= np.linalg.norm(W,2)

            # every 10 iterations, check convergence
            if i % 10 == 0:
                dist = frobenius(V, W*W.T*V)
                convgraph[i/10] = dist

                if pdist - dist < self.stopconv:
                    converged = 1
                    break

                pdist = dist

        return NMFResult((W,), convgraph, dist, converged)
