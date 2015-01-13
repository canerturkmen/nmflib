"""
Module for performing clustering on the Million Song Dataset - Genre data, as presented in
http://labrosa.ee.columbia.edu/millionsong/blog/11-2-28-deriving-genre-dataset

Author: Caner Turkmen <caner.turkmen@boun.edu.tr>

"""

import os,sys
sys.path.append("/Users/Caner/code/nmflib/")

from nmf import NMF
from projective import ProjectiveNMF
from nscsparse import NSpecSparse
from scipy import misc
import pylab as pl
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder, scale, normalize, MinMaxScaler, LabelBinarizer
from sklearn.cluster import KMeans

MSD_FILE_PATH = "/Users/Caner/code/nmflib/data/msd_genre/"


#%% import the MSD Genre Dataset
def msd():

    A1 = pd.read_csv(MSD_FILE_PATH + "msd1.txt")
    A2 = pd.read_csv(MSD_FILE_PATH + "msd2.txt")
    A = pd.concat((A1, A2))
    
    data = A.ix[:, [4,5] + range(8,34)]
    
    # generate the target variable
    le = LabelEncoder()
    labels = le.fit_transform(A.ix[:,0])
    
    # generate the time_signature and key categorical inputs w LabelBinarizer
    lb = LabelBinarizer()
    ts_data = lb.fit_transform(A.ix[:,6])
    key_data = lb.fit_transform(A.ix[:,7])
    
    print le.classes_
    ret_data = np.hstack((np.matrix(data), np.matrix(ts_data), np.matrix(key_data)))    
   
    return ret_data, labels 


X, y = msd()

#%%
np.save("/Users/Caner/code/nmflib/data/msd_genre/Xarr", X)
np.save("/Users/Caner/code/nmflib/data/msd_genre/yarr", y)
