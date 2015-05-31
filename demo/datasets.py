"""
A utility module for loading experimental datasets with sklearn.datasets like
interface
"""

from sklearn import datasets
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_wine():
    """
    Load the Wine data set from UCI repository
    """

    data = np.genfromtxt(os.path.dirname(__file__) + "/data/wine.data", delimiter=",")
    y = data[:,0].astype('int')
    X = data[:,1:]

    return {"data": X, "target": y}

def load_msdgenre():
    """
    Load the Million Song Dataset genre view
    """

    data = np.load(os.path.dirname(__file__) + "/../data/msd_genre/Xarr.npy")
    y = np.load(os.path.dirname(__file__) + "/../data/msd_genre/yarr.npy")

    # take random sample of 2% from the data
    n, m = data.shape
    samp_arr = np.random.rand(n) > .98
    X = data[samp_arr,:]

    # scale to unit interval
    d = (X - X.min(0)) / (X.max(0) - X.min(0))

    return {"data": d, "target": y[samp_arr]}

def load_ecoli():
    """
    Load the ecoli dataset from UCI ML Repository

    Reference:
     "Expert Sytem for Predicting Protein Localization Sites in Gram-Negative
     Bacteria", Kenta Nakai & Minoru Kanehisa, PROTEINS: Structure, Function,
     and Genetics 11:95-110, 1991.
    """
    dset = pd.read_csv("nmflib/demo/data/ecoli.data.txt", delim_whitespace=True, header=None)

    X = np.array(dset.ix[:,1:7])

    y = np.array(dset.ix[:,8])

    target = LabelEncoder().fit_transform(y)

    return {"data": X, "target": target}

def load_yeast():
    """
    Load the yeast dataset from UCI ML repository

    Reference:
    "A Probablistic Classification System for Predicting the Cellular
           Localization Sites of Proteins", Paul Horton & Kenta Nakai,
           Intelligent Systems in Molecular Biology, 109-115.
	   St. Louis, USA 1996.
    """

    dset = pd.read_csv("nmflib/demo/data/yeast.data.txt", delim_whitespace=True, header=None)

    X = np.array(dset.ix[:,1:5])
    X2 = np.array(dset.ix[:,7:8])

    y = np.array(dset.ix[:,9])

    target = LabelEncoder().fit_transform(y)

    d = np.c_[X, X2]

    return {"data": d, "target": target}

def load_digits():
    """
    wrapper for the scikit-learn digits data set
    """
    return datasets.load_digits()

def load_iris():
    """
    wrapper for the Iris data set
    """
    return datasets.load_iris()
