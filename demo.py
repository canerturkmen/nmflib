"""
Demo for NMF and PNMF in the library
"""

# add current directory to PYTHONPATH
import os,sys
sys.path.append("/Users/Caner/code/nmflib/")

from nmf import NMF
from projective import ProjectiveNMF
from scipy import misc
import pylab as pl
import numpy as np

#%% --
# 1. Lena

# train
lena = misc.lena()

result_pnmf = ProjectiveNMF(lena, 75).predict()
w = result_pnmf.matrices[0]
lena_hat_pnmf = w * w.T * lena

result_nmf = NMF(lena, 75, objective="kl").predict()
lena_hat_nmf = np.dot(result_nmf.matrices[0], result_nmf.matrices[1])

#%% show results

pl.figure(1)
pl.subplot(131)
pl.title("Original")
pl.imshow(lena, cmap="gray")

pl.subplot(132)
pl.title("NMF")
pl.imshow(lena_hat_nmf, cmap="gray")

pl.subplot(133)
pl.title("PNMF")
pl.imshow(lena_hat_pnmf, cmap="gray")


pl.figure(2)
pl.subplot(211)
pl.title("NMF")
pl.plot(result_nmf.convgraph)
pl.subplot(212)
pl.title("PNMF")
pl.plot(result_pnmf.convgraph)

pl.show()
#%% --
# 2. Wine Dataset (UCI Repository)



