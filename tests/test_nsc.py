"""
Tests for nonnegative spectral clustering
"""

from ..nsc import NSpecClus
import numpy as np


class TestNSC:

    def __init__(self):
        from sklearn.datasets import load_iris

        iris = load_iris()
        X, y = iris["data"], iris["target"]

        self.X = X


    def test_nsc_converge_nn(self):
        # Test that the objective function is non-increasing


        self.nsc = NSpecClus(self.X, 5, stopconv=1e-4)
        self.res = self.nsc.predict()

        cg = self.res.convgraph

        print cg
        assert not np.any((np.roll(cg, 1) - cg)[3:] > 0)


    def test_nsc_converge_gauss(self):
        # Test that the objective function is non-increasing


        self.nsc = NSpecClus(self.X, 5, stopconv=1e-4, affinity="gaussian")
        self.res = self.nsc.predict()

        cg = self.res.convgraph

        print cg
        assert not np.any((np.roll(cg, 1) - cg)[3:] > 0)


    # def test_pnmf_objective(self):
    #     # Test that the objective function is calculated correctly
    #
    #     W = self.res.matrices[0]
    #
    #     assert np.linalg.norm(self.X - W*W.T*self.X, 'fro') - self.res.objvalue < 1e-10
