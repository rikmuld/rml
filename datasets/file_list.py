import os

import torch.utils.data as data

from PIL import Image
from typing import Callable


class FileList(data.Dataset):
    def __init__(self, root: str, image_root: str, file_list: str,
                 ext: str = None, train: bool = True,
                 transform: Callable = None,
                 target_transform: Callable = None):

        self.root = root
        self.image_root = image_root
        self.file_list = file_list
        self.ext = ext
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
    def setup(self):
        data_list = os.path.join(self.root, self.file_list)
        
        self.data = [file.strip() for file in open(data_list)]
    
    def class_to_idx(self, clss: str):
        raise NotImplementedError
    
    def idx_to_class(self, idx: int):
        raise NotImplementedError
    
    def __getitem__(self, index: int):
        sample = self.data[index]
        
        if self.ext:
            sample = sample + "." + self.ext
        
        path = os.path.join(self.image_root, sample)
        
        y = sample.split("/")[-2]
        y = self.class_to_idx(y)
        
        x = Image.open(path)
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y
        
    def __len__(self):
        return len(self.data)
