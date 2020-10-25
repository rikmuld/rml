import torch

import numpy as np
import os
import torch.utils.data

from typing import List
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from collections import Counter
from itertools import accumulate
from PIL import Image

from . import utils
from . import generic as gutils


def unit(x):
    return x


class PreLoader:
    def __init__(self, dl: DataLoader, device=utils.device):
        xs, ys = [], []

        for i, data in enumerate(dl):
            xs.append(data[0].to(device))
            ys.append(data[1].to(device))

        self.input = xs
        self.target = ys

    def __iter__(self):
        order = np.random.permutation(len(self))

        for i in range(len(self)):
            yield self.input[order[i]], self.target[order[i]]

    def __len__(self):
        return len(self.input)


class SimpleDS:
    def __init__(self, x, y, transform = unit, transform_y = unit):
        self.x = x
        self.y = y
        self.transform = transform
        self.transform_y = transform_y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx: int):
        return self.transform(self.x[idx]), self.transform_y(self.y[idx])


class ImageTask(Dataset):
    def __init__(self, root, images, ext, transform):
        self.root = root
        self.X = images
        self.ext = ext
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        x = os.path.join(self.root, str(x)) + "." + self.ext
        x = Image.open(x).convert("RGB")
        
        return self.transform(x)


class ImageTaskWT(ImageTask):
    def __init__(self, root, images, targets, ext, x_transform, y_transform=unit):
        super(ImageTaskWT, self).__init__(root, images, ext, x_transform)
        
        self.y = targets
        self.y_transform = y_transform
    
    def __getitem__(self, idx):
        x = super(ImageTaskWT, self).__getitem__(idx)
        y = self.y_transform(self.y[idx])
        
        return x, y
    
def build_balanced_sampler(labels, dataset_size=None):
    if dataset_size is None:
        dataset_size = len(labels)

    weights_per_class = [1/x for x in Counter(labels).values()]
    weights_per_example = [weights_per_class[c] for c in labels]

    return WeightedRandomSampler(weights_per_example, dataset_size, replacement=True)

def relative_size_to_actual(sizes, data_size):
    total_splits = sum(sizes)
    return list(map(lambda i: int((i / total_splits) * data_size), sizes)) 

def random_splits(sizes, data_size):
    return random_splits_do(relative_size_to_actual(sizes, data_size))

def random_splits_do(splits):
    slices = list(accumulate(splits))
    slices = list(zip([0] + slices[:-1], slices[:-1] + [-1]))    
    slices = list(map(lambda i: slice(i[0], i[1]), slices))
                
    idxs = np.random.permutation(sum(splits))

    return list(map(lambda s: idxs[s], slices))

def random_splits_apply(splits, seq):
    return [seq[split] for split in splits]


def balanced_random_splits(sizes, labels, equal_size_per_class_dim_0=False):
    if len(sizes) != 2 and equal_size_per_class_dim_0:
        raise NotImplementedError

    idxmap = gutils.build_idx_map(labels)
    idxmap = {l:np.array(idxs) for l, idxs in idxmap.items()}
    
    if not equal_size_per_class_dim_0:
        splits = [random_splits_apply(random_splits(sizes, len(ids)), ids) for ids in idxmap.values()]
    else:
        idx_values = idxmap.values()

        splits = [relative_size_to_actual(sizes, len(ids)) for ids in idx_values]
        dim_0_size = min([x[0] for x in splits])
        splits = [[dim_0_size, len(ids) - dim_0_size - 1] for ids in idx_values]
        splits = [random_splits_do(split) for split in splits]
        splits = [random_splits_apply(split, ids) for split, ids in zip(splits, idx_values)]

    splits = list(zip(*splits))
    splits = [np.random.permutation(np.concatenate(split)) for split in splits]
    
    return splits