
from sklearn.datasets import load_iris
from .. import SparseNMF, NMF
import numpy as np

iris = load_iris()
X, y = iris["data"], iris["target"]

nmf = NMF(X, 5)
nmf_res = nmf.predict()

def test_sparse_fails_bad_solver():
    try:
        SparseNMF(X, 5, eta=5, beta=4, solver="badmedicine")
    except Exception as e:
        assert type(e) == ValueError

def test_nmf_correctness():
    # Test that the update rule is indeed non-increasing
    cg = nmf_res.convgraph

    assert not np.any((np.roll(cg, 1) - cg)[1:] < 1e5)

def test_nmf_convergence():
    # Test that the update rule is indeed non-increasing
    if nmf_res.converged:
        cg = nmf_res.convgraph
        x = cg[cg > 0]

        assert x[-2] - x[-1] < nmf.stopconv
    else:
        return True
