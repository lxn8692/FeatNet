import torch.nn as nn
import torch


class FmFMTransform(nn.Module):
    def __init__(self, featureNumb, dim):
        super(FmFMTransform, self).__init__()
        self.dim = dim
        self.featureNumb = featureNumb
        temp = nn.Parameter(torch.zeros(size=(featureNumb, featureNumb, dim, dim)))
        torch.nn.init.normal_(temp.data, mean=0, std=0.001)
        # self.act = torch.nn.LeakyReLU()
        self.matrix = temp
        self.out = None

    # [batch,feature,dim]-> [B,F,1,1,Dim] * [1,F,F,D,D]
    def forward(self, inputFeature: torch.Tensor):
        _input = inputFeature[:, :, None, None, :]
        mask = torch.triu(torch.ones(self.featureNumb, self.featureNumb, device=_input.device, requires_grad=False), 1)[
               :, :, None, None]
        matrix = torch.mul(mask, self.matrix)
        weight = matrix[None, :, :, :, :]
        trans = torch.matmul(_input, weight)
        # trans = self.act(trans)
        ori = inputFeature[:, None, :, :, None]
        # print('ori\n',ori)
        result = torch.matmul(trans, ori).squeeze()
        # print('result\n',result.detach().numpy())
        self.out = torch.sum(result, dim=2)
        # print('self.out',self.out)
        return self.out


class FvFMTransform(nn.Module):
    def __init__(self, featureNumb, dim):
        super(FvFMTransform, self).__init__()
        self.dim = dim
        self.featureNumb = featureNumb
        temp = nn.Parameter(torch.zeros(size=(featureNumb, featureNumb, dim)))
        torch.nn.init.normal_(temp.data, mean=0, std=0.001)
        # self.act = torch.nn.LeakyReLU()
        self.matrix = temp
        self.out = None

    # [batch,feature,dim]-> [B,F,1,Dim] * [1,F,F,D]
    def forward(self, inputFeature: torch.Tensor):
        _input = inputFeature[:, :, None, :]
        mask = torch.triu(torch.ones(self.featureNumb, self.featureNumb, device=_input.device, requires_grad=False), 1)[
               :, :, None]
        matrix = torch.mul(mask, self.matrix)
        weight = matrix[None, :, :, :]
        trans = torch.mul(_input, weight)[:, :, :, None, :]
        # trans = self.act(trans)
        ori = inputFeature[:, None, :, :, None]
        # print('ori\n',ori)
        result = torch.matmul(trans, ori).squeeze()
        # print('result\n',result.detach().numpy())
        self.out = torch.sum(result, dim=2)
        # print('self.out',self.out)
        return self.out


class FM2LinearTransform(nn.Module):
    def __init__(self, featureNumb, dim):
        super(FM2LinearTransform, self).__init__()
        self.dim = dim
        self.featureNumb = featureNumb
        self.matrix = nn.Parameter(torch.zeros(size=(featureNumb, dim)))
        torch.nn.init.normal_(self.matrix.data, mean=0, std=0.01)
        self.out = None

    # [batch,feature,dim]->[b,f,1,d]*[1,f,d,1]
    def forward(self, inputFeature: torch.Tensor, inputWeight=None):
        _in = inputFeature[:, :, None, :]
        if inputWeight is None:
            _weight = self.matrix[None, :, :, None]
        else:
            _weight = inputWeight[None, :, :, None]
        self.out = torch.matmul(_in, _weight).squeeze()
        return self.out


class FwFMTransform(nn.Module):
    def __init__(self, featureNumb, dim):
        super(FwFMTransform, self).__init__()

    def forward(self, inputFeature: torch.Tensor):
        pass


class FvElementWiseTransform(nn.Module):
    def __init__(self, featureNumb, dim):
        super(FvElementWiseTransform, self).__init__()
        self.dim = dim
        self.featureNumb = featureNumb
        temp = nn.Parameter(torch.zeros(size=(featureNumb, featureNumb, dim)))
        torch.nn.init.normal_(temp.data, mean=0, std=0.001)
        self.nonZeroMask = [[i, j] for i in range(featureNumb - 1) for j in range(i + 1, featureNumb)]
        # self.act = torch.nn.LeakyReLU()
        self.matrix = temp
        self.out = None

    # [batch,feature,dim]-> [B,F,1,Dim] * [1,F,F,D]
    def forward(self, inputFeature: torch.Tensor):
        _input = inputFeature[:, :, None, :]
        mask = torch.triu(torch.ones(self.featureNumb, self.featureNumb, device=_input.device, requires_grad=False), 1)[
               :, :, None]

        matrix = torch.mul(mask, self.matrix)
        weight = matrix[None, :, :, :]
        trans = torch.mul(_input, weight)
        # trans = self.act(trans)
        ori = inputFeature[:, None, :, :]
        # print('ori\n',ori)
        result = torch.mul(trans, ori)
        # print('result\n',result.detach().numpy())
        self.out = None
        # print('self.out',self.out)
        return self.out


if __name__ == '__main__':
    loss = torch.nn.BCELoss()
    model = FmFMTransform(3, 3)
    label = torch.ones((1))
    for i in range(100):
        input = torch.arange(9).reshape((1, 3, 3)).type(dtype=torch.float)
        out = model(input)
        ls = loss(out, loss)
