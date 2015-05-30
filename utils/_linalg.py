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

def normalize_matrix_columns(W):
    """
    Normalize a matrix columns to unit norm

    :param W: matrix to be normalized
    :type W: numpy.array
    :returns: normalized matrix
    """

    # TODO: Zero columns!!
    return W / np.linalg.norm(W, 2, 0)

def matrix_approx_equal(W, H):
    """
    If two matrices are approximately equal, return True, else False. Utility function
    required in tests.
    """
    if W.shape == H.shape:
        return np.linalg.norm(W - H, 'fro') < 1e-4
    else:
        raise ValueError("Matrix dimensions must match")

def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes

    :param clusters: the cluster assignments array
    :type clusters: numpy.array

    :param classes: the ground truth classes
    :type classes: numpy.array

    :returns: the purity score
    :rtype: float
    """

    A = np.c_[(clusters,classes)]

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]
