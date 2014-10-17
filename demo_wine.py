"""
A demo on the UCI Repository Wine dataset
"""
import numpy as np

def wine(path="/Users/Caner/code/nmflib/data/wine.data"):
    A = np.loadtxt(path,delimiter=",")
    return A[:,1:], A[:,0].astype('int')



