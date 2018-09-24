from copy import deepcopy
from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, model_params):
        self.model_params = deepcopy(model_params)

    @abstractmethod
    def _get_layer_from_param(self, param):
        raise NotImplementedError()

    def _get_model(self, params):
        layers = []
        for param in params:
            layers.append(self._get_layer_from_param(param))
        self.layers = nn.ModuleList(layers)

    def __call__(self, x, training=True):
        if training:
            self.train()
        else:
            self.eval()
        for layer in self.layers:
            x = layer(x)
        return x