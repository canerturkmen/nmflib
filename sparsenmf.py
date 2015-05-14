
from .nmf import BaseNMF
from scipy.optimize import nnls
from scipy.sparse import issparse

class SparseNMF(BaseNMF):
    """

    This class solves the Sparse-NMF formulation by Alternating Least Squares (NLS).

    For solving the optimization problem, it relies on SciPy's NNLS solver.

    """

    def __init__(self, X, k, **kwargs):
        """

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

    def predict():

        m, n = self.X.shape

        # don't work with sparse matrices
        if issparse(self.X):
            X = self.X.todense()
        else:
            X = self.X

        pdist = sys.maxint #very large number

        # Initialize W and H with random values
        W = np.random.rand(m, self.k)
        H = np.random.rand(self.k, n)

        for i in xrange(m):
            W[i, :] = nnls(H, self.X[i, :])
        for j in xrange(m):
            H[:, j] = nnls()


    def _update_scipy_nnls(W, H):
        pass

    def _update_scipy_lstsq(W, H):
        pass
