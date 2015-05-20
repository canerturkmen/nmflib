"""
A wrapper for performing clustering with the NMF variants found in
the library
"""

from .. import NMF, SparseNMF, ClusterNMF, NSpecClus, ProjectiveNMF
import numpy as np


class NMFClustering:
    """

    Perform clustering with one of the NMF algorithms in the database. Initializer
    takes one keyword argument with the following options:

    - "nmf": Perform simple (classical) NMF with Multiplicative Update rules,
        reducing Euclidean error (Lee and Seung, 2001)
    - "sparse": Perform Sparse-NMF with Alternating Nonnegative Least Squares
        (Kim and Park)
    - "projective": Projective NMF with multiplicative update rules (Yang and Oja)
    - "cluster": "Cluster-NMF" (Ding et al), a symmetricity constrained NMF formulation
        on Convex-NMF
    - "spectral": Nonnegative Spectral Clustering with Normalized Cuts (Ding et al)


    The class provides a scikit-learn like interface, with the constructor setting up parameters
    and the `fit_predict` function taking care of the algorithm run.

    """

    def __init__(self, n_clusters=10, algorithm="nmf", options={}, maxiter=1000, stopconv=.001):
        """

        """

        self.k = n_clusters
        self.nmf_class = None
        self.kwargs = {} # keyword arguments that will be passed to NMF constructor

        filter_dict = lambda x, ls : {k: x[k] for k in ls if x.get(k)}

        if algorithm == "nmf":
            # perform normal NMF
            self.nmf_class = NMF

        elif algorithm == "sparse":
            self.nmf_class = SparseNMF
            # select relevant kwargs
            self.kwargs = filter_dict(options, ("eta", "beta", "solver"))

        elif algorithm == "spectral":
            self.nmf_class = NSpecClus
            self.kwargs = filter_dict(options, ("affinity", "gamma", "nn"))

        elif algorithm == "projective":
            self.nmf_class = ProjectiveNMF

        elif algorithm == "cluster":
            self.nmf_class = ClusterNMF

        else:
            raise ValueError("Unrecognized algorithm provided")

        self.kwargs.update({"maxiter": maxiter, "stopconv": stopconv})

    def fit_predict(self, X):
        """
        Fit a clustering algorithm on the data provided in X. The data is expected
        to be of shape (n_features, n_samples).

        :param X: a data matrix of shape (n_features, n_samples)
        :returns: (y, result) a single array containing cluster identifiers, and the NMFResult object
        :rtype: tuple
        """

        nmf = self.nmf_class(X, self.k, **self.kwargs) # set up the algorithm

        result = nmf.predict()

        W = result.matrices[0]
        W = np.nan_to_num(W)
        # W /= np.linalg.norm(W, 2, 0)

        return np.argmax(result.matrices[0], 1), result
