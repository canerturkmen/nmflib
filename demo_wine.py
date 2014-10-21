"""
A demo on the UCI Repository Wine dataset
"""
import os,sys
sys.path.append("/Users/Caner/code/nmflib/")

from nmf import NMF
from projective import ProjectiveNMF
from scipy import misc
import pylab as pl
import numpy as np


def wine(path="/Users/Caner/code/nmflib/data/wine.data"):
    A = np.loadtxt(path,delimiter=",")
    return A[:,1:], A[:,0].astype('int')

#%%

wine_data, wine_labels = wine()

#%%

pnmf = ProjectiveNMF(wine_data, 15)

result = pnmf.predict()

pl.plot(result.convgraph)