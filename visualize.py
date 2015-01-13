# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:01:07 2015

@author: Caner
"""

import os,sys
sys.path.append("/Users/Caner/code/nmflib/")

from sklearn.decomposition import PCA
#import pylab as pl


import numpy as np
import matplotlib as pl
pl.use('TkAgg')

#%%

X, y = msd()

# take only dance, jazz, soul into account
nz = nonzero(np.any((y==2, y==5, y==9), axis=0))
X, y = X[nz], y[nz]
#%% USPS version

# after running demo_usps.py

X, y = data, labels


#%% USPS version
pca = PCA(n_components=5)

X_trans = pca.fit_transform(X)

data = np.hstack((X_trans, np.matrix(y).T))

#%%



np.random.shuffle(data)
sample = data[:500,:]

#%%
pl.figure()


for i in range(5):
    for j in range(5):
        try:
            #pl.subplot(5,5,((j)*5)+i)
            pl.figure()
            pl.scatter(sample[:,i].A1, sample[:,j].A1,  20, sample[:,5].A1)
            pl.show()
        except IndexError:
            print i,j

