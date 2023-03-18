import torch
import torch.nn as nn


class BiLinearLayer(nn.Module):
    def __init__(self, featureNumb, embedSize):
        super(BiLinearLayer, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.weight = nn.Parameter(torch.zeros(featureNumb, embedSize, embedSize))
        nn.init.normal(self.weight, mean=0, std=0.01)

    def forward(self, feature):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        biLinear = torch.matmul(torch.unsqueeze(feature, 2), torch.unsqueeze(self.weight, 0))
        # biLinear = feature
        interaction = biLinear * torch.unsqueeze(feature, 1)
        return interaction


class BiLinearDotLayer(nn.Module):
    def __init__(self, featureNumb, embedSize):
        super(BiLinearDotLayer, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.weight = nn.Parameter(torch.zeros(featureNumb, embedSize, embedSize))
        nn.init.normal(self.weight, mean=0, std=0.01)

    def forward(self, feature):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        biLinear = torch.matmul(torch.unsqueeze(feature, 2), torch.unsqueeze(self.weight, 0))
        # biLinear = feature
        interaction = biLinear * torch.unsqueeze(feature, 1)
        sum = torch.sum(interaction, dim=-1)
        return sum


class BiLinearLayerV2(nn.Module):
    def __init__(self, featureNumb, embedSize):
        super(BiLinearLayerV2, self).__init__()
        self.embedSize = embedSize
        self.featureNumb = featureNumb
        self.weight = nn.Parameter(torch.zeros(featureNumb, embedSize, embedSize))
        self.weightLeft = nn.Parameter(torch.zeros(featureNumb, featureNumb))
        nn.init.normal(self.weight, mean=0, std=0.01)
        nn.init.normal(self.weightLeft, mean=0, std=0.01)

    def forward(self, feature):
        # [B,F,1,D][1,F,D,D]-> [b,f,d]
        biLinear = torch.matmul(torch.unsqueeze(feature, 2), torch.unsqueeze(self.weight, 0))
        interaction = biLinear * torch.unsqueeze(feature, 1)
        output = interaction * self.weightLeft[None, :, :, None]
        return output


class BiLinearFieldLayer(nn.Module):
    def __init__(self, featureNumb):
        super(BiLinearFieldLayer, self).__init__()
        xList = []
        yList = []
        for i in range(featureNumb):
            for j in range(i + 1, featureNumb):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)
        self.out = None

    # [batch,feature,dim]-> [B,F,1,Dim] * [1,F,F,D]
    def forward(self, inputFeature: torch.Tensor, fieldVec: torch.Tensor):
        _input = inputFeature[:, None, :, None, :]
        matrix = torch.matmul(fieldVec[None, :, :, None], fieldVec[:, None, None, :])
        weight = matrix[None, :, :, :]
        trans = torch.matmul(_input, weight).squeeze(dim=3)
        ori = inputFeature[:, None, :, :]
        # print('ori\n',ori)
        result = trans * ori
        self.out = result
        return self.out
