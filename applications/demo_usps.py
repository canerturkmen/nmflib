"""
Projective NMF demo with the usps data set (handwritten digits)
"""

from nmf import NMF
from projective import ProjectiveNMF
from scipy import misc
from nsc import NSpecClus
from nscsparse import NSpecSparse
import pylab as pl
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def usps(path="/Users/Caner/code/nmflib/data/usps/"):

    train = np.loadtxt(path+"zip.train")
    test = np.loadtxt(path+"zip.test")
    A = np.vstack((train,test))

    return A[:,0].astype('int'), A[:, 1:]

def show_digit(arr):
    arr_n = np.reshape(arr, (16,16))
    pl.imshow(arr_n, cmap="gray")


labels, data = usps()

data = data + np.ones(np.shape(data))

#%%

#from experiment import Experiment


exp = Experiment(data, labels, 10)

exp_res = exp.run()

print(exp_res)

#%%

nsckm = NSpecSparseKM(data, 10, maxiter=2000)

res = nsckm.predict()