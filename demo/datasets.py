"""
A utility module for loading experimental datasets with sklearn.datasets like
interface
"""

from sklearn import datasets

def load_wine():
    """
    Load the Wine data set from UCI repository
    """
    pass

def load_usps():
    """
    Load the USPS handwritten digits data set
    """
    pass

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
