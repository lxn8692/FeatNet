import torch
import torch.nn as nn
from Models.BaseTransforms.STRTransform import STRTransform


class FeatureSTRLayer(nn.Module):
    def __init__(self, featureNumb, featureDim, init=-15, field=None):
        super(FeatureSTRLayer, self).__init__()
        self.featureDim = featureDim
        self.featureNumb = featureNumb
        if field is None:
            self.field = nn.Parameter(init * torch.ones(featureNumb, featureDim))
        else:
            self.field = field
        self.STR = STRTransform()

    # [B,F,D]
    def forward(self, feature):
        output = self.STR(feature, self.field[None, :, :])
        return output
