import torch

import numpy as np
import torch.utils.data

from typing import List
from torch.utils.data import DataLoader
from itertools import accumulate

from . import utils


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
    def __init__(self, x, y = None, transform = None):
        self.x = x
        self.transform = transform

        if y is not None:
            self.y = y

        if transform is not None:
            self.transform = transform

        if y is None and transform is None:
            self.get_item_fn = self.get_x
        elif y is None and transform is not None:
            self.get_item_fn = self.get_tx
        elif y is not None and transform is None:
            self.get_item_fn = self.get_xy
        else:
            self.get_item_fn = self.get_txy
    
    def get_x(self, idx):
        return self.x[idx]

    def get_tx(self, idx):
        return self.transform(self.x[idx])

    def get_xy(self, idx):
        return self.x[idx], self.y[idx]

    def get_txy(self, idx):
        return self.transform(self.x[idx]), self.y[idx]
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.get_item_fn(idx)
    

def random_splits(sizes: List[int], *seqs):
    data_size = len(seqs[0])
    total_splits = sum(sizes)
    
    idxs = np.random.permutation(data_size)

    slices = list(map(lambda i: int((i / total_splits) * data_size), sizes)) 
    slices = list(accumulate(slices))
    slices = list(zip([0] + slices[:-1], slices[:-1] + [-1]))    
    slices = list(map(lambda i: slice(i[0], i[1]), slices))
                
    return list(map(lambda s: [seq[idxs[s]] for seq in seqs], slices))
