# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:40:30 2014

@author: Caner

Codefile for data exploration performed on the msd genre data set
"""
import os,sys
sys.path.append("/home/ubuntu/")

import logging
logging.basicConfig(filename='log/app.2.log',level=logging.DEBUG)

import numpy as np
import pylab as pl
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler

#%%

MSD_FILE_PATH = "/Users/Caner/code/nmflib/data/msd_genre/"

X = np.load(MSD_FILE_PATH + "Xarr.npy")
y = np.load(MSD_FILE_PATH + "yarr.npy")


#%%

def draw_all_vars(X):
    """
    Draw a histogram of all the variables in the matrix X

    :param X: input matrix
    :type X: numpy.ndarray array-like
    """
    a = X.shape[1]
    for i in range(a):
        print "Now input var: %s" % i
        pl.figure()
        pl.hist(X[:,i], bins=100)
        pl.show()

#%%

#draw_all_vars(X)

#%% The preprocessing step shall do the following:

# 1. Take negative of var 0 - loudness
# 2. Take minmax scale of all

X[:,0] = -X[:,0]
XX_MinMax = MinMaxScaler().fit_transform(X)

#draw_all_vars(XX_MinMax)
