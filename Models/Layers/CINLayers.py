import torch
import torch.nn as nn
from Models.BaseTransforms.CINTransform import CINTransform


class CINLayer(nn.Module):
    def __init__(self, featureNumb, embedSize, headNum=2, depth=3):
        super(CINLayer, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.headNum = headNum
        self.depth = depth
        self.linear = nn.Parameter(torch.zeros(headNum * depth, 1))
        nn.init.normal(self.linear, mean=0, std=0.01)
        cin = []
        for i in range(depth):
            if i == 0:
                cin.append(CINTransform(headNum, featureNumb, featureNumb))
            else:
                cin.append(CINTransform(headNum, headNum, featureNumb))
        self.CIN = nn.ModuleList(cin)
        self.output = None

    def forward(self, feature):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        input1 = feature
        input2 = feature
        result = []
        for i in range(self.depth):
            input1 = self.CIN[i](input1, input2)
            result.append(torch.sum(input1, dim=2, keepdim=True))
        result = torch.cat(result, dim=1)
        self.output = result.transpose(1, 2) @ self.linear[None, :, :]
        return self.output.squeeze(1)
