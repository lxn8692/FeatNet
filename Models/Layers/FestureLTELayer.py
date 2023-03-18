import torch
import torch.nn as nn
from Models.BaseTransforms.LTE import LTETransform
from torch.autograd import Variable


class FeatureLTELayer(nn.Module):
    def __init__(self, featureNumb, featureDim, init=4, field=None):
        super(FeatureLTELayer, self).__init__()
        self.featureDim = featureDim
        self.featureNumb = featureNumb
        if field is None:
            self.field = nn.Parameter(torch.ones(featureNumb, featureDim))
            nn.init.normal(self.field, mean=init, std=0.01)
        else:
            self.field = field
        self.LTE = LTETransform()

    # [B,F,D]
    def forward(self, feature):
        indicator = torch.tanh(self.field)
        step = self.LTE.apply(indicator)
        output = feature * step[None, :, :]
        return output
