from . import file_list as data
from ..common.downloadable import Downloadable

import os
import glob
import PIL.Image as Image

from typing import Callable


class Food101(Downloadable, data.FileList):    
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    
    def __init__(self, root: str, train: bool = True,
                 transform: Callable = None, target_transform: Callable = None,
                 download: bool = False):

        Downloadable.__init__(self, root, Food101.url, download)
        data.FileList.__init__(self,
                               root=root,
                               image_root=os.path.join(root, "images"),
                               file_list=Food101.get_file_listing(train),
                               ext="jpg",
                               transform=transform,
                               target_transform=target_transform)
        
        Downloadable.setup(self)        
        
    def download_finished(self):
        self.fix_image_mode()
            
    def fix_image_mode(self):
        images = glob.glob(os.path.join(self.root, 'images/*/*'))

        for file in images:
            image = Image.open(file)

            if str(image.mode) == "L":        
                new_image = image.convert("RGB")
                new_image.save(file)

    def _setup(self):
        classes = os.path.join(self.root, "meta/classes.txt")
        labels = os.path.join(self.root, "meta/labels.txt")

        self._classes = [clss.strip() for clss in open(classes)]
        self._labels = [label.strip() for label in open(labels)]

        self.clss_to_idx = {clss:idx for idx, clss in enumerate(self.classes)}
        self.idx_to_clss = {idx:clss for idx, clss in enumerate(self.classes)}

        data.FileList.setup(self)
    
    def class_to_idx(self, clss: str):
        return self.clss_to_idx[clss]
    
    def idx_to_class(self, idx: int):
        return self.idx_to_clss[idx]
    
    @property
    def labels(self):
        return self._labels
        
    @property
    def classes(self):
        return self._classes
        
    def exists(self):
        if not os.path.exists(self.root):
            return False
        
        files = os.listdir(self.root)
                
        if not "meta" in files or not "images" in files:
            return False
        
        images = os.listdir(os.path.join(self.root, "images"))
        meta = os.listdir(os.path.join(self.root, "meta"))
                
        if not len(images) >= 101 or not len(meta) >= 6:
            return False
        
        return True
    
    @staticmethod
    def get_file_listing(train: bool):
        if train:
            return "meta/train.txt"
        else:
            return "meta/test.txt"
