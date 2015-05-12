# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 13:03:24 2014

@author: Caner Turkmen

Functions for calculating the distance between two matrices
"""
import numpy as np
import sys

def frobenius(A, B):
    """
    Function for calculating the Euclidean distance between two matrices
    """
    return np.linalg.norm(A-B, 'fro')

def kldivergence(A,B):
    """
    Function for determining the divergence of A from B
    as presented in Lee and Seung, otherwise known as generalized Kullback-Leibler
    divergence or I-divergence.

    :param A: first matrix
    :type A: numpy.ndarray
    :param B: second matrix
    :type B: numpy.ndarray

    :returns: the divergence
    :rtype: float
    """
    return np.sum((A*np.log(A/B) - A + B))
