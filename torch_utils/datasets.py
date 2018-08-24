import torch
import torch.utils.data as tdata
import numpy as np


class NumpyDataset(tdata.Dataset):
    '''Subclass of torch.utils.data.Dataset from numpy array

    X; array-like
    y: array-like, optional
    '''
    def __init__(self, X, y=None):
        self.X = torch.tensor(X).float()
        if y is not None:
            if np.issubdtype(y.dtype, np.integer):
                y = torch.tensor(y).long()
            else:
                y = torch.tensor(y).float()
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index]
        else:
            return self.X[index], self.y[index]