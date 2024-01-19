"""
PyKMTools
=====

PyKMTools: A python tool base for kannmu
Figure drawing tools

Author: Kannmu
Date: 2024/1/6
License: MIT License
Repository: https://github.com/Kannmu/PyKMTools
"""
import numpy as np
import PyKMTools.Data as d
import matplotlib.pyplot as plt
import seaborn as sn

# plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = ["Times New Roman"]

def PlotOneD(X, SavePath:str = None,Title:str = "Title", XLabel:str = "X", YLabel:str = "Y"):
    plt.clf()
    X = d.toarray(X)
    if(len(X.shape) != 1):
        raise Exception("PyKMTools: Drawing Error: Input data is not 1-dim")
    plt.subplots()
    plt.plot(range(len(X)),X)
    # Set Figure Style
    plt.title(Title)
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)

    if (SavePath != None):
        plt.savefig(SavePath, dpi = 200)
        plt.close()
    else:
        plt.show()

def PlotTwoCurve(Input, SavePath:str = None, Title:str = "Title", XLabel:str = "X", YLabel:str = "Y", InputLabels:list = []):
    plt.clf()
    Input = d.toarray(Input)
    
    if(len(Input.shape) == 1):
        raise Exception("PyKMTools: Drawing Error: Input data is 1-dim, please use Draw.PlotAndSaveOneD instead")
    
    if len(Input[0]) != len(Input[1]):
        raise ValueError('The two curves do not have the same length.')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    plt.title(Title)
    ax1.set_xlabel(XLabel)

    color = 'tab:blue'
    ax1.set_ylabel(YLabel, color=color)
    ax1.plot(Input[0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel(YLabel, color=color)
    ax2.plot(Input[1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if InputLabels:
        ax1.legend([InputLabels[0]], loc='upper left')
        ax2.legend([InputLabels[1]], loc='upper right')
    if (SavePath != None):
        plt.savefig(SavePath, dpi = 200)
        plt.close()
    else:
        plt.show()

def HeatMapAndSave(Input, SavePath:str,Title:str = "Title", XLabel:str = "X", YLabel:str = "Y"):
    plt.clf()
    plt.subplots()
    sn.heatmap(Input, annot=True)

    # Set Figure Style
    plt.title(Title)
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.savefig(SavePath, dpi=200)
    plt.close()