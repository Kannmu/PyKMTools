"""
PyKMTools
=====

PyKMTools: A python tool base for kannmu

OneDArray Tools

Author: Kannmu
Date: 2024/1/4
License: MIT License
Repository: https://github.com/Kannmu/PyKMTools

"""

# Import necessary modules
import numpy as np
import scipy
import torch as t

def toarray(X:np.ndarray or list or t.Tensor, Norm = False, Stand = False):
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
    if isinstance(X, list):
        X = np.array(X)
    elif isinstance(X, t.Tensor):
        X = X.numpy()
    if not isinstance(X, np.ndarray):
        raise TypeError('Input data type not supported. It should be one of the following: np.ndarray, list, torch.Tensor')
    try:
        X = np.array(X)
        if(Norm):
            X = norm(X)
        if(Stand):
            X = stand(X)
    except Exception as e:
        raise Exception(f"PyKMTools: toarray Error: {str(e)}")
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
    return np.asarray(Freq), np.asarray(FFT_Vector)

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

def ReverseData(X:np.ndarray or list or t.Tensor, Axis:int = 0):
    if isinstance(X, list):
        X = np.array(X)
    elif isinstance(X, t.Tensor):
        X = X.numpy()
    if not isinstance(X, np.ndarray):
        raise TypeError('Input data type not supported. It should be one of the following: np.ndarray, list, torch.Tensor')

    if Axis >= X.ndim:
        raise ValueError(f'Invalid axis: {Axis}. Axis should be less than the number of dimensions in X, which is {X.ndim}.')
    
    X_reversed = np.flip(X, axis=Axis)
    return X_reversed

def ReSample(X:np.ndarray or list, Rate:float = 1.0, Axis:int = 0):
    """
    Resample the input data along the specified axis

    Parameters
    ----------
    X: np.ndarray or list
        Input data, can be of type numpy array or list

    Rate: float, optional
        The percentage of the original data to be resampled, default is 0.9

    Axis: int, optional
        The axis along which the data is to be resampled, default is 0

    Returns
    ----------
    X_resampled: np.ndarray
        The resampled data
    """

    if isinstance(X, list):
        X = np.array(X)
    elif isinstance(X, t.Tensor):
        X = X.numpy()

    if not isinstance(X, np.ndarray):
        raise TypeError('Input data type not supported. It should be one of the following: np.ndarray, list, torch.Tensor')

    if Axis >= X.ndim:
        raise ValueError(f'Invalid axis: {Axis}. Axis should be less than the number of dimensions in X, which is {X.ndim}.')
    
    length = X.shape[Axis]
    resample_length = int(length * Rate)

    start_index = np.random.choice(length - resample_length + 1)
    slice_obj = [slice(None)]*X.ndim
    slice_obj[Axis] = slice(start_index, start_index + resample_length)

    X_resampled = X[tuple(slice_obj)]
    # print(start_index,slice_obj,X_resampled)
    return X_resampled

