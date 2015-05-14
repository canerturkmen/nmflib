
from .nmf import BaseNMF, NMFResult
from .utils import frobenius
from scipy.optimize import nnls
from scipy.sparse import issparse
import numpy as np

class SparseNMF(BaseNMF):
    """


    """

    def __init__(self, X, k, **kwargs):
        """
        Initialize the Sparse-NMF class, an implementation of the cost function formulation given by
        Kim and Park.

        ```
        SparseNMF(X, k, eta=1, beta=1, solver="nnls")
        ```

        .. math::
            \min_{W,H} \frac{1}{2} \left[ ||A - WH^T ||_F^2 + \eta ||W||_F^2 + \beta \sum_{j=1}^{n} ||H(j,:)||_1^2  \right] s.t. W,H > 0

        :param X: The data matrix, **must** be a data matrix of shape (n_samples, n_features)
        :param k: number of clusters

        :return:

        """
        BaseNMF.__init__(self, X, k, **kwargs)

        # Initialize regularization parameters
        self.eta = kwargs.get("eta") or 1
        self.beta = kwargs.get("beta") or 1

        self.solver_param = kwargs.get("solver") or "nnls"

        if self.solver_param == "nnls":
            self.solver = self._update_scipy_nnls
        elif self.solver_param == "lstsq":
            self.solver = self._update_scipy_lstsq
        else:
            raise ValueError("Unrecognized solver given")

    def predict(self):

        m, n = self.X.shape

        # don't work with sparse matrices
        if issparse(self.X):
            X = self.X.todense()
        else:
            X = self.X

        pdist = 1e10 #very large number

        # Initialize W and H with random values
        W = np.random.rand(m, self.k)
        H = np.random.rand(self.k, n)

        dist = 0
        converged = False
        convgraph = np.zeros(self.maxiter / 10)

        for i in xrange(self.maxiter):

            W, H = self.solver(W, H)

            if i % 10 == 0:
                dist = frobenius(X, np.dot(W,H))
                convgraph[i/10] = dist

                if pdist - dist < self.stopconv:
                    converged = True
                    break

                pdist = dist

        # normalize W,H before returning
        norms = np.linalg.norm(W, 2, axis=0)
        norm_gt_0 = norms > 0
        W[:, norm_gt_0] /= norms[norm_gt_0]
        H[norm_gt_0, :] = ((H[norm_gt_0, :].T) * norms[norm_gt_0]).T

        return NMFResult((W,H), convgraph, dist, converged)


    def _update_scipy_nnls(self, W, H):
        """
        Run the update step with regularized cost function, and Nonnegative Least Squares
        provided by SciPy (activeset variant)

        :param W: the left factorizing matrix
        :param H: the right factorizing matrix
        :type W: numpy.ndarray
        :type H: numpy.ndarray
        :returns: two-tuple (W,H) with new matrices
        """

        # 'augmented' data matrix with vector of zeros
        Xaug = np.r_[self.X, np.zeros((1,H.shape[1]))]

        # 'augmented' left factorizing matrix with vector of ones
        Waug = np.r_[W, np.sqrt(self.beta)*np.ones((1,5))]

        Htaug = np.r_[H.T, np.sqrt(self.eta) * np.eye(H.shape[0])]
        Xaugkm = np.r_[self.X.T, np.zeros(W.T.shape)]

        for i in xrange(W.shape[0]):
            W[i, :] = nnls(Htaug, Xaugkm[:, i])[0]

        for j in xrange(H.shape[1]):
            H[:, j] = nnls(Waug, Xaug[:, j])[0]

        return (W,H)

    def _update_scipy_lstsq(W, H):
        pass