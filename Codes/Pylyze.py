"""
Pylyze
=====

Pylyze: A data processing tool for python

Author: Kannmu
Date: 2024/1/4
License: MIT License
Repository: https://github.com/Kannmu/Pylyze

"""

# Import necessary modules
import numpy as np
import sys
import warnings

import scipy

def toarray(X, Norm = False, Stand = False):
    """
    Transfer a data to numpy array
    
    Parameters
    ----------
    X: array-like
        Input data
    
    Returns
    ----------
    X: numpy array
        data in numpy array type
    """
    try:
        X = np.array(X)
        if(Norm):
            X = norm(X)
        if(Stand):
            X = stand(X)
    except Exception as e:
        raise Exception(f"Pylyze Error: {str(e)}")
    return X

def norm(X):
    """
    Data normalization, map data into range of [0,1]
    
    Parameters
    ----------
    X: array-like
        Input data
    
    Returns
    ----------
    X: numpy array
        normalized data
    """
    X = toarray(X)
    if np.max(X) == np.min(X):
        return X
    else:
        _range = np.max(X) - np.min(X)
        return (X - np.min(X)) / _range

def stand(X):
    """
    Data standardization, map data into distribution of mean = 0, std = 1
    
    Parameters
    ----------
    X: array-like
        Input data
    
    Returns
    ----------
    X: numpy array
        Standardized data
    """
    X = toarray(X)
    mu = np.mean(X)
    sigma = np.std(X)
    return (X - mu) / (sigma + 1e-20)

def Amp_FFT(X, SampleRate, HighCutFreq = 100):
    """
    Amplitude Fast Fourier Transform
    
    Parameters
    ----------
    X: array-like
        Input data
    SampleRate: float
        Sample rate of input data
    HighCutFreq: float
        High Cut of Frequency For Results
    Stand: bool
        Enable standardization for input data
    
    Returns
    ----------
    Freq: numpy array
        FFT frequency, works as X Axis
    FFT_Vector: numpy array
        Amplitude data for all frequencies
    """
    X = toarray(X)
    FFT_Vector = np.abs(np.fft.fft(X)) / len(X)
    Freq = np.fft.fftfreq(len(FFT_Vector), d=1 / SampleRate)
    FFT_Vector = FFT_Vector[: int(len(FFT_Vector) / 2)]
    Freq = Freq[: int(len(Freq) / 2)]
    Freq_Index = np.where(Freq <= HighCutFreq)
    Freq = Freq[Freq_Index]
    FFT_Vector = FFT_Vector[: len(Freq)]
    return Freq, FFT_Vector

def BandPassFilter(X, SampleRate, lowpass, highpass, FilterLevel):
    """
    Butterworth band pass filter
    
    Parameters
    ----------
    X: array-like
        Input data
    SampleRate: float
        Sample rate of input data
    lowpass: float
        lowpass frequency of band
    highpass: float
        highpass frequency of band
    FilterLevel: int
        Filter degree
        
    Returns
    ----------
    Freq: numpy array
        FFT frequency, works as X Axis
    FFT_Vector: numpy array
        Amplitude data for all frequencies
    """
    X = toarray(X)
    b, a = scipy.signal.butter(
        FilterLevel,
        [
            ((lowpass) / (SampleRate / 2)),
            ((highpass) / (SampleRate / 2)),
        ],
        "bandpass",
    )
    Temp = scipy.signal.filtfilt(b, a, X)

    return Temp

# For Test Usage
if __name__ == "__main__":
    X = range(10) + np.sin(range(10))
    print("Origin: ",X)
    print("Norm: ",norm(X))
    print("Stand: ",stand(X))
    print("Amp_FFT: ",Amp_FFT(X,1,Stand=True))