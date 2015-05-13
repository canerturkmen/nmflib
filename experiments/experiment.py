# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 22:41:17 2014

@author: Caner

A module for running experiments with several clustering alternatives and reporting on the results.

The techniques used are KMeans from sklearn, classical NMF, and Sparse NN NSC
"""
#%%
from nscsparse import NSpecSparse
from nspeckm import NSpecSparseKM
from nmf import NMF
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmiscore

class Experiment:
    """
    An experiment object encapsulates one experimental setting. (One data set, and one K setting).
    The class, when initialized, saves a copy of the data and the K setting. When run() is invoked,
    the object runs the experiments, saves the results and reports on the nmi results.
    """

    def __init__(self, X, y, k):
        """
        :param X: the training data
        :param y: the original classes
        :param k: the number of clusters
        """

        self.k = k
        self.X = X
        self.y = y

    def run(self):
        """
        :returns: Tuple, such as (<dict of nmis, with algorithm names as keys >)
        """

        nsc = NSpecSparse(self.X, self.k, maxiter=2000)
        nmf = NMF(self.X, self.k)
        km = KMeans(n_clusters=self.k)
        nsckm = NSpecSparseKM(self.X, self.k, maxiter=2000)

        nsc_result = nsc.predict()
        nmf_result = nmf.predict()
        km_result  = km.fit_predict(self.X)
        nsckm_result = nsckm.predict()

        w_nsc = nsc_result.matrices[0].todense()
        w_nmf = nmf_result.matrices[0]
        w_nsckm = nsckm_result.matrices # gets only the labels

        arrays = {
            'nsc': np.array(np.argmax(w_nsc, axis=1))[:,0],
            'nmf': np.array(np.argmax(w_nmf, axis=1)),
            'km': km_result,
            'nsckm': w_nsckm
        }

        nmi = {k: nmiscore(arrays[k], self.y) for k in arrays.keys()}

        return (nmi, arrays)
