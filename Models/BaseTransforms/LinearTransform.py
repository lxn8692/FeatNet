import torch
from collections import OrderedDict
import torch.nn as nn


class LinearTransform(nn.Module):
    def __init__(self, layerDimension: list, dropLast=False, act=None):
        super(LinearTransform, self).__init__()
        layers = []
        self.layerDimensions = layerDimension
        for i in range(0, len(layerDimension) - 2):
            temp = nn.Linear(layerDimension[i], layerDimension[i + 1], bias=True)
            nn.init.normal(temp.weight, mean=0,std=0.01)
            layers.append((f'linear{i}', temp))
            layers.append((f'drop_out{i}',nn.Dropout(p=0.1)))
            if act is None:
                layers.append((f'Relu{i}', nn.ReLU(inplace=False)))
            else:
                layers.append((str(act), act))

        temp = nn.Linear(layerDimension[-2], layerDimension[-1], bias=True, )
        nn.init.xavier_normal(temp.weight, gain=1.414)
        layers.append((f'linear{-1}', temp))
        layers.append((f'drop_out{-1}', nn.Dropout(p=0.1)))
        if not dropLast:
            layers.append((f'Relu{-1}', nn.ReLU(inplace=False)))

        self.layers = nn.Sequential(OrderedDict(layers))
        self.output = None

    def forward(self, featureVec):
        self.output = self.layers(featureVec)
        return self.output
