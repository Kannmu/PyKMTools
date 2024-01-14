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
import PyKMTools.OneDArray as oned
import matplotlib.pyplot as plt
import seaborn as sn

def PlotAndSaveOneD(X, SavePath:str, Title:str = "Title", XLabel:str = "X", YLabel:str = "Y"):
    plt.clf()
    X = oned.toarray(X)
    if(len(X.shape) != 1):
        raise Exception("PyKMTools: Drawing Error: Input data is not 1-dim")
    plt.subplots()
    plt.plot(range(len(X)),X)
    plt.title(Title)
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.savefig(SavePath, dpi = 200)

def PlotAndSaveND(Input, SavePath:str, Title:str = "Title", XLabel:str = "X", YLabel:str = "Y", InputLabels:list = [], YLim:list = None):
    plt.clf()
    Input = oned.toarray(Input)
    if(len(Input.shape) == 1):
        raise Exception("PyKMTools: Drawing Error: Input data is 1-dim, please use Draw.PlotAndSaveOneD instead")
    
    plt.subplots()
    plt.plot(range(Input.shape[1]), Input,label = InputLabels)
    if(YLim != None):
        plt.ylim(YLim)
    
    plt.title(Title)
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.savefig(SavePath, dpi = 200)

def HeatMapAndSave(Input, SavePath:str,Title:str = "Title", XLabel:str = "X", YLabel:str = "Y"):
    plt.clf()
    plt.subplots()
    sn.heatmap(Input, annot=True)
    plt.title(Title)
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.savefig(SavePath, dpi=200)