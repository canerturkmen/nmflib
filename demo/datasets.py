"""
A utility module for loading experimental datasets with sklearn.datasets like
interface
"""

from sklearn import datasets
import numpy as np
import os

def load_wine():
    """
    Load the Wine data set from UCI repository
    """

    data = np.genfromtxt(os.path.dirname(__file__) + "/data/wine.data", delimiter=",")
    y = data[:,0].astype('int')
    X = data[:,1:]

    return {"data": X, "target": y}

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
