# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:41:45 2014

@author: Caner

Script for running set of experiments with different transformations of data
"""
import numpy as np
from experiment import Experiment
from sklearn.preprocessing import scale, normalize, MinMaxScaler

np.seterr(all='warn')

# MSD_FILE_PATH = "/home/ubuntu/"
MSD_FILE_PATH = "/Users/Caner/code/nmflib/data/msd_genre/"

X = np.load(MSD_FILE_PATH + "Xarr.npy")
y = np.load(MSD_FILE_PATH + "yarr.npy")


XX_MinMax = MinMaxScaler().fit_transform(X)
XX_Square = scale(X)**2
XX_scaleclip = scale(X).clip(min=0)
XX_clip = normalize(X.clip(min=0)).clip(min=0)

XX_ = [XX_MinMax, XX_Square, XX_scaleclip, XX_clip]
#%%


for xx in XX_:
    exp = Experiment(X, y, 10)
    exp_res = exp.run()
    print exp_res
