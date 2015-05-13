# -*- coding: utf-8 -*-

class BaseNMF:
    """
    Class that serves as the base-class for different implementations of NMF.
    """

    maxiter = 10000
    stopconv = 1e-4

    def __init__(self, X, k, **kwargs):
        """
        Initialize the NMF problem with a matrix.

        The initializer also responds to the following keyword arguments:

        - metric: (str) the objective metric to be reduced. Not all NMF versions have implementations of different metrics. May
            take "eu" (default) for euclidean distance or "kl" for generalized Kullback-Leibler divergence (I-divergence)
        - maxiter: (int) the maximum number of iterations to be performed. Default is 10000
        - stopconv: (int) the convergence criterion for the objective function. Default is 40

        :param X: matrix
        :param k: number of dimensions for NMF
        """

        #TODO: ClusterNMF can have negative entries!
        if X.min() < 0:
            raise Exception("The matrix cannot have negative entries")

        if kwargs.get("maxiter"):
            self.maxiter = kwargs.get("maxiter")

        if kwargs.get("stopconv"):
            self.stopconv = kwargs.get("stopconv")

        self.X = X
        self.k = k

class NMFResult:
    """
    Simple object for storing the results of an NMF training run
    """

    convgraph = None # an array of objective function values to plot convergence
    matrices = None # a python **list** of factorizing matrices
    objvalue = None # the final value of the objective function
    converged = None

    def __init__(self, matrices, convgraph=None, objvalue=None, converged=None):
        self.matrices = matrices
        self.convgraph = convgraph
        self.objvalue = objvalue
        self.converged = converged
