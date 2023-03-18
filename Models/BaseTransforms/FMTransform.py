import torch.nn as nn
import torch


class FMTransform(nn.Module):
    def __init__(self):
        super(FMTransform, self).__init__()

    # [batch,feature,dim]-> [B,F,Dim] * [B,Dim,F]->[B,F,F]->Sum(upper)
    def forward(self, inputFeature: torch.Tensor):
        print('inputFeature\n', torch.isnan(inputFeature).any())
        trans = torch.matmul(inputFeature, torch.transpose(inputFeature, -1, -2))
        print('trams\n', trans.cpu().detach().numpy())
        triUpper = torch.triu(trans, 1)
        result = torch.sum(triUpper, dim=2)
        return result


class FmLinearTransform(nn.Module):
    def __init__(self, featureNumb):
        super(FmLinearTransform, self).__init__()
        self.linear = nn.Parameter(torch.zeros(size=(1, featureNumb)))
        nn.init.normal_(self.linear, mean=0, std=0.01)

    def forward(self):
        return self.linear
