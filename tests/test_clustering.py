from ..cluster import NMFClustering
from .. import *

# test that the NMF classes are assigned correctly

def test_clusternmf_accept_nmf():
    obj = NMFClustering(5, "nmf")
    assert obj.nmf_class.__name__ == "NMF"

def test_clusternmf_k():
    obj = NMFClustering(5, "nmf")
    assert obj.k == 5

def test_clusternmf_accept_nsc():
    obj = NMFClustering(5, "spectral")
    assert obj.nmf_class.__name__ == "NSpecClus"

def test_clusternmf_accept_pnmf():
    obj = NMFClustering(5, "projective")
    assert obj.nmf_class.__name__ == "ProjectiveNMF"
