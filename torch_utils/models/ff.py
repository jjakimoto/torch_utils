from copy import deepcopy

import torch.nn as nn

from ..layers import Flatten
from .core import BaseModel


class FeedForward(BaseModel):
    def __init__(self, model_params, *args, **kwargs):
        self.model_params = deepcopy(model_params)

    def _get_layer_from_param(self, param):
        param = deepcopy(param)
        name = param['name']
        del param['name']
        if name == 'conv1d':
            layer = nn.Conv1d(**param)
        elif name == 'conv2d':
            layer = nn.Conv2d(**param)
        elif name == 'conv3d':
            layer = nn.Conv3d(**param)
        elif name == 'flatten':
            layer = Flatten()
        elif name == 'linear':
            layer = nn.Linear(**param)
        elif name == 'dropout':
            layer = nn.Dropout(**param)
        elif name == 'relu':
            layer = nn.ReLU()
        elif name == 'sigmoid':
            layer = nn.Sigmoid()
        elif name == 'tanh':
            layer = nn.Tanh()
        else:
            raise NotImplementedError(f'No Implementation for name={name}')
        return layer
