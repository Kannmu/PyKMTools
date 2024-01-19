"""
PyKMTools
=====

PyKMTools: A python tool base for kannmu

Data Argument Tools

Author: Kannmu
Date: 2024/1/19
License: MIT License
Repository: https://github.com/Kannmu/PyKMTools

"""

import torch as t
import numpy as np
import scipy

def AddNormalNoise(X:np.ndarray or list, NoiseAmp:float):
    """
    Add normal noise to the input data

    Parameters
    ----------
    X: np.ndarray or list
        Input data, can be of type numpy array or list

    NoiseAmp: float
        Noise amplification factor, determines the intensity of the noise added

    Returns
    ----------
    X: numpy array
        Data after adding noise
    """
    X = np.array(X)
    Noise = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Noise[i, :] = (
            NoiseAmp * np.random.rand()
            * np.random.normal(0, X[i, :].std(), X.shape[1])
        )
    return X + Noise

