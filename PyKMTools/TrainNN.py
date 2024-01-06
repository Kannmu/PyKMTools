"""
PyKMTools
=====

PyKMTools: A python tool base for kannmu
Train Neural Network for Classification

Author: Kannmu
Date: 2024/1/4
License: MIT License
Repository: https://github.com/Kannmu/PyKMTools
"""
import math
import os
import shutil
import sys
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from torch.utils import data
from torch.utils.data import DataLoader, random_split

import PyKMTools.OneDArray as OneD


class TrainProcess:
    def __init__(
        self,
        Total_Epoch=400,
        Learning_Rate=0.0001,
        Batch_Size=32,
        Validate_Per_N_Epoch=3,
        Dropout_Rate=0.5,
        Train_Rate=0.9,
        ModelSavePath="./Model/Default",
        DataProcessingPath="./__init__.py",
    ) -> None:
        """
        Initialize the training process

        Parameters
        ----------
        EPOCH: int
            Hyperparameter, total epochs to train
        Learning_Rate: float
            Hyperparameter, Learning Rate
        Validate_Per_N_Epoch: int
            Hyperparameter, Perform validation after N epochs
        Dropout_Rate: float
            Hyperparameter, Dropout Rate
        Train_Rate: float
            Data parameter, Train Data Rate
        """
        self.InitCheck()
        self.ModelSaveFolderInit(ModelSavePath, DataProcessingPath)

        self.Total_Epoch = Total_Epoch
        self.Learning_Rate = Learning_Rate
        self.Batch_Size = Batch_Size
        self.Validate_Per_N_Epoch = Validate_Per_N_Epoch
        self.Dropout_Rate = Dropout_Rate
        self.Train_Rate = Train_Rate

        self.LogList = []

    def InitCheck(self):
        print("\n###### Check environment ######")
        print("torch.__version__ = ", torch.__version__)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        print("torch.cuda.is_available = ", torch.cuda.is_available())

        print("torch.cuda.device_count = ", torch.cuda.device_count(), "\n")
        if not torch.cuda.is_available():
            raise Exception("No GPU detected")
        
    def CreateDirs(self,ModelSavePath):
        os.makedirs(ModelSavePath)
        os.makedirs(ModelSavePath + "Code/")
        os.makedirs(ModelSavePath + "Log/")
        os.makedirs(ModelSavePath + "Model/")
        os.makedirs(ModelSavePath + "Figure/")

    def ModelSaveFolderInit(self, ModelSavePath, DataProcessingPath):
        try:
            self.CreateDirs(ModelSavePath)
        except Exception as e:
            # print(e)
            YESORNOT = input(
                str(ModelSavePath) + "is already exist, overlap it? [yes/no]"
            )
            if YESORNOT == "yes":
                print("Files in ", ModelSavePath, " deleted")
                shutil.rmtree(ModelSavePath)
                self.CreateDirs(ModelSavePath)
            elif YESORNOT == "no":
                ModelSavePath = input("Input another model save path")
            else:
                raise Exception("Wrong Input")

        CodeSavePath = str(DataProcessingPath.split("/")[-1])
        # print(CodeSavePath)
        shutil.copy(
            DataProcessingPath, ModelSavePath + "Code/" + CodeSavePath
        )

        Temp_ACC = 0
        Temp_Loss = np.inf

        try:
            F = open("./" + ModelSavePath + "Log/" + "Log.txt", "r")
            Line = F.readline()
            if Line != "":
                print(Line)
                Temp_ACC = float(Line[9:20])
                Temp_Loss = float(Line[38:47])
                print("Highest Acc so far is: ", Temp_ACC)
                print("Lowest Loss so far is: ", Temp_Loss)
            else:
                Temp_ACC = 0.0
                Temp_Loss = np.inf
            F.close()
        except Exception as e:
            F = open("./" + ModelSavePath + "Log/" + "Log.txt", "w+")
            Temp_ACC = 0.0
            Temp_Loss = np.inf
            F.close()
            print("Create new log file")

    class Dataset(data.Dataset):
        """
        Create Dataset Class for Training and Validate data
        """

        def __init__(self, Process_Ins):
            self.Inputs, self.Targets = Process_Ins.Total_Inputs , Process_Ins.Total_Targets

        def __getitem__(self, index):
            Target = self.Targets[index]
            Input = self.Inputs[index]
            return Input, Target

        def __len__(self):
            return len(self.Targets)

    def LoadData(self, Inputs, Targets):
        """
        Load Array-like data from input

        Parameters
        ----------
        Inputs: NDArray-like
            Input data
        Targets: OneDArray-like
            Target Labels
        """
        try:
            self.Total_Inputs = np.asarray(Inputs)
            self.Total_Targets = np.asarray(Targets)
        except Exception as e:
            raise Exception(f"PyKMTools: LoadData Error: {str(e)}")
        
        self.TotalDataset = self.Dataset(self)

        TrainSize = int(self.Train_Rate*len(self.Total_Targets))
        TestSize = len(self.Total_Targets) - TrainSize
        self.TrainDataset, self.ValDataset = random_split(self.TotalDataset, [TrainSize, TestSize])

        self.TrainLoader = DataLoader(
            self.TrainDataset,
            batch_size=self.Batch_Size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        self.ValLoader = DataLoader(
            self.ValDataset,
            batch_size=self.Batch_Size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

    def Start(self):
        """
        Start Training Process

        """
        pass
