import PyKMTools as pk
import numpy as np


Process = pk.tnn.TrainProcess(ModelSavePath="./Model/Test/",DataProcessingPath="./UsageDemo.py")


Video = pk.vdo.Video("your_video_path/Video_0.mp4")
Video.FrameExtractor(fps=1, quality=2)
