import numpy as np
from ..utils import *

A = np.array([[1,2,3,4], [5,6,7,8]])

def test_approxequal_en2_notequal():
    B = np.random.rand(5,10)

    #add perturbation to one element
    C = B.copy()
    C[0,0] += .1

    assert not matrix_approx_equal(B,C)

def test_approxequal_en6_equal():
    B = np.random.rand(5,10)
    C = B + 1e-6

    assert matrix_approx_equal(B,C)

def test_normalize_columns():

    fix = np.array([[ 0.19611614,  0.31622777,  0.3939193 ,  0.4472136 ],
                        [ 0.98058068,  0.9486833 ,  0.91914503,  0.89442719]])
    norm = normalize_matrix_columns(A)
    assert matrix_approx_equal(norm, fix)
