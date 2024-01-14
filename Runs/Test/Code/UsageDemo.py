import PyKMTools as pk
import numpy as np
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(2,16)
        self.output = nn.Linear(16,4)
    def forward(self,X):
        X = self.fc1(X)
        X = self.output(X)


DenseModel = Model()

Hyperparameters = pk.tnn.Hyperparameters(
    N_Targets=4,
    RunSavePath="./Runs/Test/",
    DataProcessingPath="./UsageDemo.py"

)

Process = pk.tnn.TrainProcess(
    Hyperparameters = Hyperparameters, 
    Model=DenseModel, 
    Optimizer="AdamW",
    LossFunc="CrossEntropy"
)

Process.LoadData([[1,2], [2,3], [3,4], [4,5],[1.5,2.5], [2.5,3.5], [3.5,4.5], [4.5,5.5]], [0,1,2,3,0,1,2,3])

Process.StartTrain()

# Video = pk.vdo.Video("your_video_path/Video_0.mp4")
# Video.FrameExtractor(fps=1, quality=2)
