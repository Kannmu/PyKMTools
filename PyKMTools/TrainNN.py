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
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torch.utils.data import DataLoader, random_split

import PyKMTools.Draw as dr


class Hyperparameters:
    """
    Set Training Hyperparameters

    Parameters
    ----------
    Total_Epoch: int, default=400
        Hyperparameter, total epochs to train
    Learning_Rate: float, default=0.0001
        Hyperparameter, Learning Rate
    Batch_Size: int, default=32
        Hyperparameter, Batch size for training
    Validate_Per_N_Epoch: int, default=3
        Hyperparameter, Perform validation after N epochs
    Dropout_Rate: float, default=0.5
        Hyperparameter, Dropout Rate
    Train_Rate: float, default=0.9
        Data parameter, Train Data Rate
    Weight_Decay: float, default=0.01
        Hyperparameter, Weight Decay for optimizer
    N_Targets: int, default=4
        Hyperparameter, Number of targets for the model
    RunSavePath: str, default="./Runs/Default/"
        The path where the trained model will be saved
    DataProcessingPath: str, default="./UsageDemo.py"
        The path of the data processing script
    """

    def __init__(
        self,
        Total_Epoch=400,
        Learning_Rate=0.0001,
        Batch_Size=32,
        Validate_Per_N_Epoch=3,
        Dropout_Rate=0.5,
        Train_Rate=0.9,
        Weight_Decay=0.01,
        Num_Works=0,
        N_Targets=4,
        RunSavePath="./Runs/Default/",
        DataProcessingPath="./UsageDemo.py",
    ) -> None:
        self.Total_Epoch = Total_Epoch
        self.Learning_Rate = Learning_Rate
        self.Batch_Size = Batch_Size
        self.Validate_Per_N_Epoch = Validate_Per_N_Epoch
        self.Dropout_Rate = Dropout_Rate
        self.Train_Rate = Train_Rate
        self.Weight_Decay = Weight_Decay
        self.Num_Works = Num_Works

        self.N_Targets = N_Targets

        self.RunSavePath = RunSavePath
        self.DataProcessingPath = DataProcessingPath


class TrainProcess:
    def __init__(
        self, Hyperparameters: Hyperparameters, Model, Optimizer: str, LossFunc: str
    ) -> None:
        """
        Initialize the training process

        Parameters
        ----------
        Hyperparameters : PyKMTools.Hyperparameters
            An instance of the Hyperparameters class from the PyKMTools library.
            This instance should store all the hyperparameters required for the training process.

        Model : nn.Module or str
            If an instance of nn.Module, this is the model to be trained.
            If a string, this should be the name of a model architecture to be used.
            The architecture should be available in the PyTorch library.
            For example, 'resnet18' for a ResNet-18 architecture.

        Optimizer : str
            The name of the optimizer to be used for training the model.
            This should be a string corresponding to an optimizer available in the PyTorch library,
            such as 'SGD' for Stochastic Gradient Descent.

        LossFunc : str
            The loss function to be used during training.
            This should be a string corresponding to a loss function available in the PyTorch library,
            such as 'CrossEntropyLoss' for the cross-entropy loss function.
        """

        self.Total_Epoch = Hyperparameters.Total_Epoch
        self.Learning_Rate = Hyperparameters.Learning_Rate
        self.Batch_Size = Hyperparameters.Batch_Size
        self.Validate_Per_N_Epoch = Hyperparameters.Validate_Per_N_Epoch
        self.Dropout_Rate = Hyperparameters.Dropout_Rate
        self.Train_Rate = Hyperparameters.Train_Rate
        self.Weight_Decay = Hyperparameters.Weight_Decay
        self.Num_Works = Hyperparameters.Num_Works
        self.N_Targets = Hyperparameters.N_Targets

        self.RunSavePath = Hyperparameters.RunSavePath
        self.DataProcessingPath = Hyperparameters.DataProcessingPath

        self.ValLossList = []
        self.ValAccList = []

        self.TrainLossArray = np.zeros([])
        self.TrainAccArray = np.zeros([])

        self.InitCheck()

        self.RunSaveFolderInit(self.RunSavePath, self.DataProcessingPath)

        self.SetModel(Model)

        self.SetOptimizer(Optimizer)

        self.SetLossFunc(LossFunc)

    def InitCheck(self):
        print("\n###### Check environment ######")
        print("torch.__version__ = ", torch.__version__)

        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        Device_Count = torch.cuda.device_count()
        print("torch.cuda.device_count = ", Device_Count)
        print("torch.cuda.is_available = ", torch.cuda.is_available())
        if torch.cuda.is_available():
            self.Device = torch.device("cuda")
        else:
            self.Device = torch.device("cpu")
        print("torch.cuda.device_count = ", torch.cuda.device_count(), "\n")
        if not torch.cuda.is_available():
            raise Exception("No GPU detected")

    def CreateDirs(self, RunSavePath):
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
        os.makedirs(RunSavePath)
        os.makedirs(RunSavePath + "Code/")
        os.makedirs(RunSavePath + "Log/")
        self.LogPath = RunSavePath + "Log/"
        os.makedirs(RunSavePath + "Model/")
        self.ModelPath = RunSavePath + "Model/"
        os.makedirs(RunSavePath + "Figure/")
        self.FigurePath = RunSavePath + "Figure/"

    def RunSaveFolderInit(self, RunSavePath, DataProcessingPath):
        """
        Initializes the run save folder.

        Parameters:
        ----------
            RunSavePath: str
                The path to save the model. If the path already exists, the user is asked whether to overwrite it.
            DataProcessingPath: str
                The path of the data processing code. This code file will be copied into the 'Code' folder under the model save path.

        The method attempts to create a directory under the given RunSavePath. If the directory already exists, it asks the user whether to overwrite it.

        If the user chooses to overwrite, it deletes the existing directory and recreates it. If the user chooses not to overwrite, they are asked to input a new model save path.

        The method also tries to read the 'Log.txt' file in the 'Log' folder under the model save path, retrieves and prints the highest accuracy and the lowest loss so far.

        If the 'Log.txt' file does not exist, it creates a new 'Log.txt' file.
        """
        try:
            self.CreateDirs(RunSavePath)
        except Exception as e:
            print("Run path exist, overlap it")
            YESORNOT = input(
                str(RunSavePath) + "is already exist, overlap it? [yes/no]  "
            )
            if YESORNOT in ["yes", "y", "Y", "YES", ""]:
                print("Files in ", RunSavePath, " deleted")
                shutil.rmtree(RunSavePath)
                self.CreateDirs(RunSavePath)
            elif YESORNOT in ["no", "N", "n", "NO"]:
                RunSavePath = input("Input another model save path")
            else:
                raise Exception("Wrong Input")

        CodeSavePath = str(DataProcessingPath.split("/")[-1])

        shutil.copy(DataProcessingPath, RunSavePath + "Code/" + CodeSavePath)

        self.Temp_ACC = 0
        self.Temp_Loss = np.inf

        try:
            F = open("./" + self.LogPath + "Log.txt", "r")
            Line = F.readline()
            if Line != "":
                print(Line)
                self.Temp_ACC = float(Line[9:20])
                self.Temp_Loss = float(Line[38:47])
                print("Highest Acc so far is: ", self.Temp_ACC)
                print("Lowest Loss so far is: ", self.Temp_Loss)
            else:
                self.Temp_ACC = 0.0
                self.Temp_Loss = np.inf
            F.close()
        except Exception as e:
            F = open("./" + self.LogPath + "Log.txt", "w+")
            self.Temp_ACC = 0.0
            self.Temp_Loss = np.inf
            F.close()
            print("Create new log file")

    class Dataset(data.Dataset):
        """
        Create Dataset Class for Training and Validate data
        """

        def __init__(self, Process_Ins):
            self.Inputs, self.Targets = (
                Process_Ins.Total_Inputs,
                Process_Ins.Total_Targets,
            )

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
        print("Loading Data\n")
        self.LoadDataFinishFlag = False
        try:
            self.Total_Inputs = torch.tensor(
                np.asarray(Inputs), dtype=torch.float32
            ).to(self.Device)
            self.Total_Targets = torch.tensor(np.asarray(Targets), dtype=torch.long).to(
                self.Device
            )
            if self.Total_Inputs.shape[0] != self.Total_Targets.shape[0]:
                raise Exception(
                    f"PyKMTools: LoadData Error: Input Data and Labels are not in the same number"
                )

        except Exception as e:
            raise Exception(f"PyKMTools: LoadData Error: {str(e)}")

        self.TotalDataset = self.Dataset(self)

        self.TrainSize = int(self.Train_Rate * len(self.Total_Targets))
        self.ValSize = len(self.Total_Targets) - self.TrainSize
        if(self.ValSize < self.Batch_Size):
            print("ValSize less than Batch_Size, please adjust Batch_Size, Train_Rate or dataset scale")
        print("Training data size: ", self.TrainSize)
        print("Training data size: ", self.ValSize, "\n")

        self.TrainDataset, self.ValDataset = random_split(
            self.TotalDataset, [self.TrainSize, self.ValSize]
        )

        self.TrainLoader = DataLoader(
            self.TrainDataset,
            batch_size=self.Batch_Size,
            shuffle=True,
            num_workers=self.Num_Works,
            drop_last=False,
        )
        self.ValLoader = DataLoader(
            self.ValDataset,
            batch_size=self.Batch_Size,
            shuffle=True,
            num_workers=self.Num_Works,
            drop_last=False,
        )

        # self.LoadDataFinishFlag = True

    def SetModel(self, Model, ResnetChannel=3):
        if type(Model) == str:
            if Model in ["Resnet18", "resnet18", "resnet 18", "resnet-18", "Resnet-18"]:
                print("Using Resnet18 Model")
                self.Model = torchvision.models.resnet18().to(self.Device)
                num_ftrs = self.Model.fc.in_features
                self.Model.fc = nn.Sequential(nn.Linear(num_ftrs, self.N_Targets))
                self.Model.conv1 = nn.Conv2d(
                    ResnetChannel, 64, kernel_size=3, stride=2, padding=1, bias=False
                )
        else:
            print("Using Self Defined Model")
            self.Model = Model.to(self.Device)

        for m in self.Model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        for m in self.Model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def SetOptimizer(self, Optimizer):
        if type(Optimizer) == str:
            if Optimizer == "Adam":
                self.Optimizer = torch.optim.Adam(
                    self.Model.parameters(), lr=self.Learning_Rate
                )
            if Optimizer == "AdamW":
                self.Optimizer = torch.optim.AdamW(
                    self.Model.parameters(),
                    lr=self.Learning_Rate,
                    weight_decay=self.Weight_Decay,
                )
        else:
            raise Exception(f"PyKMTools: LossFunc Error: Wrong input Optimizer name")
        print("Using ", Optimizer, " Optimizer")

    def SetLossFunc(self, LossFunc):
        if type(LossFunc) == str:
            if LossFunc == "CrossEntropy":
                self.Loss_func = nn.CrossEntropyLoss()
        else:
            raise Exception(
                f"PyKMTools: LossFunc Error: Wrong input loss function name"
            )
        print("Using ", LossFunc, " loss function")

    def StartTrain(self):
        """
        Start Training Process

        """
        # while(not self.LoadDataFinishFlag):
        #     pass

        print("Start Training Process \n")

        self.Model.train()

        for Epoch in range(self.Total_Epoch):
            self.Model.train()
            Correct_All = 0
            Train_Loss_List = []
            for Step, (Train_Input, Train_Target) in enumerate(self.TrainLoader):
                Train_Output = self.Model(Train_Input)

                Loss = self.Loss_func(Train_Output, Train_Target)

                nn.utils.clip_grad_norm_(
                    self.Model.parameters(), max_norm=20, norm_type=2
                )

                self.Optimizer.zero_grad()

                Loss.backward()

                self.Optimizer.step()

                _, Prediction = torch.max(Train_Output.data, 1)
                Correct_All += torch.sum(Prediction == Train_Target.data).to(
                    torch.float32
                )
                Train_Loss_List.append(Loss.item())

                # Print Training Info
                if Step % (1) == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tAcc:{:.3f}\tCorrectNum:{:.0f}".format(
                            Epoch,
                            Step * len(Train_Input),
                            self.TrainSize,
                            100.0 * Step / len(self.TrainLoader),
                            Loss.item(),
                            float(Correct_All / self.TrainSize),
                            Correct_All,
                        )
                    )
                    # Train_Loss_List = []

            self.TrainLossArray = np.append(self.TrainLossArray, np.mean(Train_Loss_List))
            self.TrainAccArray = np.append(self.TrainAccArray, float(Correct_All / self.TrainSize))
            dr.PlotTwoCurve(
                np.array([self.TrainLossArray, self.TrainAccArray]),
                self.FigurePath + "TrainLogPlot.png",
                Title="TrainLogPlot",
                XLabel="Epoch",
                YLabel="Loss/ACC",
                InputLabels=["Loss", "Accuracy"],
            )
            torch.cuda.empty_cache()

            if Epoch % self.Validate_Per_N_Epoch == 0 and Epoch > 0:
                self.Validate()

    def Validate(self):
        Correct_All_Val = 0
        Val_Loss_List = []
        print("\nStart Validate")

        # Switch Model To Evaluation Mode
        self.Model.eval()
        CM = torch.zeros(self.N_Targets, self.N_Targets)
        for Val_Step, (Val_Data, Val_Label) in enumerate(self.ValLoader):
            
            Val_Outputs = self.Model(Val_Data)

            Val_Loss = self.Loss_func(Val_Outputs, Val_Label)
            _, Prediction = torch.max(Val_Outputs.data, 1)

            Correct_All_Val += torch.sum(Prediction == Val_Label.data).to(torch.float32)
            
            Val_Loss_List.append(Val_Loss)
            
            CM = self.Calculate_CM(Val_Outputs, Val_Label, CM)

            print(
                "Val Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tAcc:{:.3f}\tCorrectNum:{:.0f}".format(
                    Val_Step,
                    Val_Step * len(Val_Data),
                    self.ValSize,
                    100.0 * Val_Step / len(self.ValLoader),
                    torch.mean(torch.tensor(Val_Loss_List)),
                    float(Correct_All_Val / self.TrainSize),
                    Correct_All_Val,
                )
            )

        ACC = float(Correct_All_Val / self.ValSize)

        try:
            Val_Loss_Value = Val_Loss.item()
            print(
                "Val_Acc: ",
                round(ACC, 3),
                "Val_Loss: ",
                round(Val_Loss_Value, 3),
                "CorrectNum:",
                int(Correct_All_Val),
            )
        except Exception as e:
            Val_Loss_Value = 100
            print(e)
            print("Wrond Loss Value", "Val_Acc: ", ACC)

        self.ValLossList.append(Val_Loss_Value)
        self.ValAccList.append(ACC)

        # Save Loss and Acc Curve
        # print(np.array([self.LossList, self.AccList]))

        dr.PlotTwoCurve(
            np.array([self.ValLossList, self.ValAccList]),
            self.FigurePath + "ValLogPlot.png",
            Title="ValLogPlot",
            XLabel="Epoch",
            YLabel="Loss/ACC",
            InputLabels=["Loss", "Accuracy"],
        )
        self.Save_Model(ACC=ACC, ValLoss=Val_Loss_Value, CM=CM)

        print("Finish Validate \n")

    def Calculate_CM(self, preds, labels, conf_matrix):
        preds = torch.argmax(preds, 1)
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1
        return conf_matrix

    def Save_Model(self, ACC: float, ValLoss: float, CM):
        # Save Model
        if ACC > self.Temp_ACC and ValLoss < self.Temp_Loss:
            self.Temp_ACC = ACC
            self.Temp_Loss = ValLoss
            # Save The Best Model On Validation
            F = open(self.LogPath + "Log.txt", "w+")
            F.writelines(
                "Val_Acc: "
                + str(ACC)
                + " Val_Loss: "
                + str(ValLoss)
                # + " Val_Loss_List: "
                # + str(self.LossList)
            )
            F.close()
            
            try:
                # Save Model
                torch.save(self.Model, self.ModelPath + "Model.pth")
                torch.save(self.Model, self.ModelPath + "Model.h5")
            except:
                pass
            # Save CM Figure
            dr.HeatMap(CM, self.FigurePath + "Confusion Matrix.png")
