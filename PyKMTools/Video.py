"""
PyKMTools
=====

PyKMTools: A python tool base for kannmu
Video Processing Tools

Author: Kannmu
Date: 2024/1/6
License: MIT License
Repository: https://github.com/Kannmu/PyKMTools
"""
import os
import subprocess

class Video:
    def __init__(self, video_path):
        """
        Initializes a Video object with the specified video path.

        Parameters
        ----------
            video_path: str)
                The path to the video file.
        """
        self.video_path = video_path

    def FrameExtractor(self, fps = 30, quality = 100, frame_dir=None):
        """
        Splits the video into frames and saves them as JPEG images.

        Parameters
        ----------
            fps: int
                The frame rate of the extractor.
            quality: int
                The quality of the JPEG images. [0-100]
            frame_dir: str, optional 
                The directory to save the frames. Defaults to the same path of the source video.
        """
        quality = int(0.29*(100 - quality) + 2)
        filename = os.path.splitext(os.path.basename(self.video_path))[0]
        if frame_dir is None:
            frame_dir = os.path.join(os.path.dirname(self.video_path), 'Frames_' + filename)
        os.makedirs(frame_dir, exist_ok=True)

        cmd = ['ffmpeg', '-i',  self.video_path,"-f", "image2", '-r', str(fps), '-q:v', str(quality), os.path.join(frame_dir, 'frame_%d.jpg')]
        try:
            subprocess.run(cmd, check=True)
            print(f'\nFrames saved to {frame_dir}')
        except subprocess.CalledProcessError as e:
            print(f'Error: {e}')

