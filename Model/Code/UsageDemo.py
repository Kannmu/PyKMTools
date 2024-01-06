import PyKMTools as pk
import numpy as np

X = np.arange(0,100,11)
# print(pk.OneD.norm(X))

Process = pk.tc.TrainProcess(
    Train_Rate = 0.85,
    ModelSavePath="./Model/",
    DataProcessingPath="./UsageDemo.py"
    )

Process.LoadData(X,[1,2,3,4])

print(Process.Total_Targets)
