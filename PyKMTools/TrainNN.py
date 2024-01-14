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

class Hyperparameters:
    """
        Set Training Hyperparameters

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
    def __init__(self,
            Total_Epoch=400,
            Learning_Rate=0.0001,
            Batch_Size=32,
            Validate_Per_N_Epoch=3,
            Dropout_Rate=0.5,
            Train_Rate=0.9,
            Weight_Decay = 0.01,
            ModelSavePath="./Model/Default",
            DataProcessingPath="./UsageDemo.py",
        ) -> None:

        self.Total_Epoch = Total_Epoch
        self.Learning_Rate = Learning_Rate
        self.Batch_Size = Batch_Size
        self.Validate_Per_N_Epoch = Validate_Per_N_Epoch
        self.Dropout_Rate = Dropout_Rate
        self.Train_Rate = Train_Rate
        self.Weight_Decay = Weight_Decay
        self.ModelSavePath = ModelSavePath
        self.DataProcessingPath = DataProcessingPath

class TrainProcess:
    def __init__(
        self,
        Hyperparameters,
        Model,
        Optimizer,
        LossFunc
    ) -> None:
        """
        Initialize the training process

        Parameters
        ----------
        Hyperparameters: PyKMTools.Hyperparameters
            Hyperparameters class, store all hyperparameters
        Model: nn.Module or str
            Model class or use the default models by giving a string model name. Options: resnet18



        """

        self.Total_Epoch = Hyperparameters.Total_Epoch
        self.Learning_Rate = Hyperparameters.Learning_Rate
        self.Batch_Size = Hyperparameters.Batch_Size
        self.Validate_Per_N_Epoch = Hyperparameters.Validate_Per_N_Epoch
        self.Dropout_Rate = Hyperparameters.Dropout_Rate
        self.Train_Rate = Hyperparameters.Train_Rate
        self.Weight_Decay = Hyperparameters.Weight_Decay


        self.SetModel(Model)

        self.SetOptimizer(Optimizer)
        
        self.SetLossFunc(LossFunc)

        self.InitCheck()
        self.ModelSaveFolderInit(Hyperparameters.ModelSavePath, Hyperparameters.DataProcessingPath)
        self.LogList = []




    def InitCheck(self):
        print("\n###### Check environment ######")
        print("torch.__version__ = ", torch.__version__)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        print("torch.cuda.is_available = ", torch.cuda.is_available())

        print("torch.cuda.device_count = ", torch.cuda.device_count(), "\n")
        if not torch.cuda.is_available():
            raise Exception("No GPU detected")
        
    def CreateDirs(self, ModelSavePath):
        """
        Creates the necessary directories for model saving.

        Parameters:
        ----------
            ModelSavePath: str 
                The base path where the model and related data will be saved. This method will create the following subdirectories under the provided path:
            
            - 'Code/': Intended to store code files related to the model.
            - 'Log/': Intended to store log files.
            - 'Model/': Intended to store the model files.
            - 'Figure/': Intended to store any figures or images related to the model.

        This method does not return anything. It will raise an exception if the directories cannot be created, for example, due to insufficient permissions.
        """
        os.makedirs(ModelSavePath)
        os.makedirs(ModelSavePath + "Code/")
        os.makedirs(ModelSavePath + "Log/")
        os.makedirs(ModelSavePath + "Model/")
        os.makedirs(ModelSavePath + "Figure/")

    def ModelSaveFolderInit(self, ModelSavePath, DataProcessingPath):
        """
        Initializes the model save folder.

        Parameters:
        ----------
            ModelSavePath: str
                The path to save the model. If the path already exists, the user is asked whether to overwrite it.
            DataProcessingPath: str
                The path of the data processing code. This code file will be copied into the 'Code' folder under the model save path.

        The method attempts to create a directory under the given ModelSavePath. If the directory already exists, it asks the user whether to overwrite it.
        
        If the user chooses to overwrite, it deletes the existing directory and recreates it. If the user chooses not to overwrite, they are asked to input a new model save path.
        
        The method also tries to read the 'Log.txt' file in the 'Log' folder under the model save path, retrieves and prints the highest accuracy and the lowest loss so far.
        
        If the 'Log.txt' file does not exist, it creates a new 'Log.txt' file.
        """
        try:
            self.CreateDirs(ModelSavePath)
        except Exception as e:
            # print(e)
            YESORNOT = input(
                str(ModelSavePath) + "is already exist, overlap it? [yes/no] "
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
            if(self.Total_Inputs.shape[0] != self.Total_Targets.shape[0]):
                raise Exception(f"PyKMTools: LoadData Error: Input Data and Labels are not in the same number")
                
        except Exception as e:
            raise Exception(f"PyKMTools: LoadData Error: {str(e)}")
        
        self.TotalDataset = self.Dataset(self)

        TrainSize = int(self.Train_Rate*len(self.Total_Targets))
        TestSize = len(self.Total_Targets) - TrainSize
        self.TrainDataset, self.ValDataset = random_split(self.TotalDataset, [TrainSize, TestSize])

        self.TrainLoader = DataLoader(
            self.TrainDataset,
            batch_size = self.Batch_Size,
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

    def SetModel(self, Model, N_Targets = 4, ResnetChannel = 3): 
        self.N_Targets = N_Targets
        print(type(Model))
        if(type(Model) == str):
            if(Model in ["Resnet18","resnet18","resnet 18","resnet-18","Resnet-18"]):
                print("Using Resnet18 Model")
                self.Model = torchvision.models.resnet18()
                num_ftrs = self.Model.fc.in_features
                self.Model.fc = nn.Sequential(nn.Linear(num_ftrs, self.N_Targets))
                self.Model.conv1 = nn.Conv2d(ResnetChannel, 64, kernel_size = 3, stride=2, padding = 1, bias=False)
        else:
            
            print("Using Self Defined Model")
            self.Model = Model

    def SetOptimizer(self, Optimizer):
        if(type(Optimizer) == str):
            if(Optimizer == "Adam"):
                print("Using CrossEntropy Loss Function")
                self.Optimizer = torch.optim.Adam(self.Model.parameters(),lr = self.Learning_Rate)
            if(Optimizer == "AdamW"):
                print("Using CrossEntropy Loss Function")
                self.Optimizer = torch.optim.AdamW(self.Model.parameters(),lr = self.Learning_Rate, weight_decay = self.Weight_Decay)
        else:
            raise Exception(f"PyKMTools: LossFunc Error: Wrong input loss function name")

    def SetLossFunc(self, LossFunc):
        if(type(LossFunc) == str):
            if(LossFunc == "CrossEntropy"):
                print("Using CrossEntropy Loss Function")
                self.Loss_func = nn.CrossEntropyLoss()
        else:
            raise Exception(f"PyKMTools: LossFunc Error: Wrong input loss function name")

    def Start(self):
        """
        Start Training Process

        """
        self.Model.train()
        for Epoch in range(self.Total_Epoch):
            self.Model.train()
            Correct_All = 0
            Train_Loss_List = []
            for Step, (Train_Data, Train_Label) in enumerate(self.TrainLoader):
                Train_Data, Train_Label = Train_Data.cuda(), Train_Label.cuda()

                Output =  self.Model(Train_Data)

                Loss = self.Loss_func(Output, Train_Label)

                nn.utils.clip_grad_norm_( self.Model.parameters(), max_norm=20, norm_type=2)

                self.SetOptimizer.zero_grad()

                Loss.backward()

                self.SetOptimizer.step()

                _, Prediction = torch.max(Output.data, 1)
                Correct_All += torch.sum(Prediction == Train_Label.data).to(torch.float32)
                Train_Loss_List.append(Loss)

                # Print Training Info
                if Step % (1) == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.6f}\tCorrectNum:{:.0f}".format(
                            Epoch,
                            Step * len(Train_Data),
                            len(self.TrainLoader.dataset),
                            100.0 * Step / len(self.TrainLoader),
                            torch.mean(torch.tensor(Train_Loss_List)),
                            float(Correct_All / len(self.TrainLoader.dataset)),
                            Correct_All,
                        )
                    )
                    Train_Loss_List = []

            if Epoch % self.Validate_Per_N_Epoch == 0 and Epoch > 0:
                Val()
        
