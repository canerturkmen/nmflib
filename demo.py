"""
Demo for NMF and PNMF in the library
"""

from nmf import NMF
from scipy import misc
import pylab as pl

#%% --
# 1. Lena

# train
lena = misc.lena()

lena_hat = NMF(lena, 75, objective="kl").predict()

#%% show results

pl.imshow(lena, cmap="gray")
pl.figure()
pl.imshow(lena_hat, cmap="gray")

#%% --
# 2. Wine Dataset (UCI Repository)

