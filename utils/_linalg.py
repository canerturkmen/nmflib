from scipy.sparse import issparse
from numpy.core import sqrt

import numpy as np

def norm(x):
    """
    Take the Frobenius norm of matrix x

    :param x: a sparse matrix
    :type x: scipy.sparse type matrix
    :returns: the Frobenius norm
    """
    if not issparse(x):
        raise TypeError("Input is not a sparse matrix")

    return sqrt((x.data**2).sum())

def trace(x):
    """
    Take the trace (sum along diagonal) of a sparse matrix x

    :param x: a sparse matrix
    :type x: scipy.sparse type matrix
    :returns: the trace
    """
    if not issparse(x):
        raise TypeError("Input is not a sparse matrix")

    if not x.ndim == 2:
        raise ValueError("trace only handles 2-way arrays (matrices)")

    m, n = x.shape

    if not m == n:
        raise ValueError("Trace is only valid for square matrices")

    return sum(x[i,i] for i in range(m))

def normalize_factor_matrices(W, H):
    """
    Bring W to unit columns while keeping W*H constant

    :param W: left factorizing matrix
    :param H: right factorizing matrix
    :type W: numpy.ndarray
    :type H: numpy.ndarray
    :returns: tuple, two-tuple (W, H)
    """
    # normalize W,H before returning
    norms = np.linalg.norm(W, 2, axis=0)
    norm_gt_0 = norms > 0
    W[:, norm_gt_0] /= norms[norm_gt_0]
    H[norm_gt_0, :] = ((H[norm_gt_0, :].T) * norms[norm_gt_0]).T

    return (W,H)
