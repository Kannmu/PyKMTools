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
import PyKMTools.OneDArray as oned
import matplotlib.pyplot as plt
import seaborn as sn

# plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = ["Times New Roman"]

def PlotAndSaveOneD(X, SavePath:str, Title:str = "Title", XLabel:str = "X", YLabel:str = "Y"):
    plt.clf()
    X = oned.toarray(X)
    if(len(X.shape) != 1):
        raise Exception("PyKMTools: Drawing Error: Input data is not 1-dim")
    plt.subplots()
    plt.plot(range(len(X)),X)

    # Set Figure Style
    plt.title(Title)
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.savefig(SavePath, dpi = 200)
    plt.close()

def PlotAndSaveTwoCurve(Input, SavePath:str, Title:str = "Title", XLabel:str = "X", YLabel:str = "Y", InputLabels:list = []):
    plt.clf()
    Input = oned.toarray(Input)
    if(len(Input.shape) == 1):
        raise Exception("PyKMTools: Drawing Error: Input data is 1-dim, please use Draw.PlotAndSaveOneD instead")
    
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    colors = ['b','g']
    axs = [ax1,ax2]
    for i, v in enumerate(Input):
        axs[i].plot(range(len(v)), v,label = InputLabels[i],c = colors[i])
        if i == 0:
            axs[i].grid(linestyle='-.')
        # else:
            # axs[i].set_axis_off()
        axs[i].set_xlabel(XLabel)
        axs[i].set_ylabel(YLabel)
        axs[i].legend()

    # if(YLim != None):
    #     plt.ylim(YLim)

    # Set Figure Style
    
    # plt.grid(linestyle='-.')
    plt.title(Title)
    plt.savefig(SavePath, dpi = 200)
    plt.close()

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