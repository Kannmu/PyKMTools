"""
Pylyze
=====

Pylyze: A X processing tool for python

Author: Kannmu
Date: 2024/1/4
License: MIT License

"""

# Import necessary modules
import numpy as np
import sys
import warnings


def toarray(X, Norm = False, Stand = False):
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
    X = toarray(X)
    if np.max(X) == np.min(X):
        return X
    else:
        _range = np.max(X) - np.min(X)
        return (X - np.min(X)) / _range

def stand(X):
    X = toarray(X)
    mu = np.mean(X)
    sigma = np.std(X)
    return (X - mu) / (sigma + 1e-20)

def Amp_FFT(X, SampleRate, HighCutFreq = 100, Stand = False):
    X = toarray(X, Stand = Stand)
    FFT_Vector = np.abs(np.fft.fft(X)) / len(X)
    Freq = np.fft.fftfreq(len(FFT_Vector), d=1 / SampleRate)
    FFT_Vector = FFT_Vector[: int(len(FFT_Vector) / 2)]
    Freq = Freq[: int(len(Freq) / 2)]
    Freq_Index = np.where(Freq <= HighCutFreq)
    Freq = Freq[Freq_Index]
    FFT_Vector = FFT_Vector[: len(Freq)]
    return Freq, FFT_Vector


# For Test Usage
if __name__ == "__main__":
    X = range(10) + np.sin(range(10))
    print("Origin: ",X)
    print("Norm: ",norm(X))
    print("Stand: ",stand(X))
    print("Amp_FFT: ",Amp_FFT(X,1,Stand=True))