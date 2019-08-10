import os
import re
import gc
import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image


def get_frames(dirname):
    """
        return: list of image files
    """
    from torchvision.datasets.folder import is_image_file
    def get_numeric(file):
        """ 
            sorting frames by frame number
        """
        return int(re.findall('\d+', os.path.split(file)[-1])[0])
    return sorted([os.path.join(dirname, file) 
                   for file in os.listdir(dirname) 
                   if is_image_file(file)], key=get_numeric)

def read_video(dirname, start_pts=0, end_pts=None):
    """
        return: video (Tensor[T, H, W, C]): the `T` video frames
                audio: ? #todo
                info : fps ...
    """
    frames = get_frames(dirname)
    video = []
    for i, frame in enumerate(frames):
        with open(frame, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.asarray(img)
            video.append(img)
            
    if end_pts is None:
        end_pts = len(video)

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))
        
    video = np.asarray(video)
    video = torch.tensor(video)
    audio = torch.tensor([]) #tmp
    info = {'video_fps': 15.0,
           'body_keypoint': None}
    sample = (video, audio, info)
    return read_video_as_clip(sample, start_pts, end_pts)


def read_video_as_clip(sample, start_pts, end_pts):
    """
        slice a video into small clips
    """
    video, audio, info = sample
    video = video[start_pts:end_pts+1]
    return (video, audio, info)


def read_video_timestamps(dirname):
    """ tmp function """
    frames = get_frames(dirname)
    return (list(range(len(frames)*1)), 15.0)
