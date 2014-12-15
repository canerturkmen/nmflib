# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:41:45 2014

@author: Caner

Script for running set of experiments with different transformations of data
"""
import os,sys
sys.path.append("/home/ubuntu/")

import logging
logging.basicConfig(filename='log/app.1.log',level=logging.DEBUG)


import numpy as np
from experiment import Experiment
from sklearn.preprocessing import scale, normalize, MinMaxScaler, LabelBinarizer


logging.info('starting execution of msdscript2')


np.seterr(all='warn')

#MSD_FILE_PATH = "/home/ubuntu/"
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
    logging.info('starting experiment')
    exp = Experiment(xx, y, 10)
    exp_res = exp.run()
    logging.info(exp_res)

#%%

# TODOS for the project:
# LabelBinarize : key, time_signature

lb = LabelBinarizer()

x1_binarized = lb.fit_transform(X[:,1])

