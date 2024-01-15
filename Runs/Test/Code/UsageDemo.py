import sys
import PyKMTools as pk
import numpy as np
import torch.nn as nn
import torch
import torchvision

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(2, 128)
        self.output = nn.Linear(128, 4)
    def forward(self,X):
        X = self.fc1(X)
        X = self.output(X)
        return X


if __name__=="__main__":
    
    DenseModel = Model()


    # Model = torchvision.models.resnet18()
    # num_ftrs = Model.fc.in_features
    # Model.fc = nn.Sequential(nn.Linear(num_ftrs, 4))
    # Model.conv1 = nn.Conv2d(4, 64, kernel_size = 3, stride=2, padding = 1, bias=False)

    Hyperparameters = pk.tnn.Hyperparameters(
        Total_Epoch=100,
        Batch_Size=8,
        N_Targets=4,
        Num_Works=0,
        RunSavePath="./Runs/Test/",
        DataProcessingPath="./UsageDemo.py"
    )

    Process = pk.tnn.TrainProcess(
        Hyperparameters = Hyperparameters, 
        Model = DenseModel, 
        Optimizer="AdamW",
        LossFunc="CrossEntropy"
    )

    X = [[i,2*i] for i in np.arange(0,100,0.1)]
    # print(np.asarray(X).shape)

    Y = [int((j[0]+j[1])%4) for j in X]
    
    # print(Y)

    # sys.exit(0)

    Process.LoadData(X, Y)

    Process.StartTrain()

    # Video = pk.vdo.Video("your_video_path/Video_0.mp4")
    # Video.FrameExtractor(fps=1, quality=2)
