
from sklearn.datasets import load_iris
from ..sparsenmf import SparseNMF

iris = load_iris()
X, y = iris["data"], iris["target"]


def test_sparse_fails_bad_solver():
    try:
        SparseNMF(X, 5, eta=5, beta=4, solver="badmedicine")
    except Exception as e:
        assert type(e) == ValueError