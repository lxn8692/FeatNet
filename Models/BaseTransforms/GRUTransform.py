import torch
import torch.nn as nn

from Utils.HyperParamLoadModule import HyperParam


class GruTransform(nn.Module):
    def __init__(self, inputSize, hiddenSize, batchFirst=True, dropout=0):
        super(GruTransform, self, ).__init__()
        # self.linear = nn.Parameter(torch.ones(size=(inputSize, hiddenSize)))
        # self.act = torch.nn.LeakyReLU()
        # nn.init.normal(self.linear, mean=0, std=0.01)
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.Gru = nn.GRU(input_size=inputSize, hidden_size=hiddenSize,
                          batch_first=batchFirst, dropout=dropout)
        nn.init.xavier_normal(self.Gru.weight_ih_l0, gain=1.414)
        nn.init.xavier_normal(self.Gru.weight_hh_l0, gain=1.414)
        self.linear2 = nn.Linear(15, 1)
        self.linear = nn.Linear(inputSize, hiddenSize)
        self.out = None

    def forward(self, feature: torch.Tensor, resultLen: torch.Tensor):
        # batchSize = feature.shape[0], maxLen = feature.shape[1]
        # idx = torch.arange(maxLen, dtype=torch.long).to(resultLen.device)
        # resultLen = resultLen.type(torch.int).to(device=resultLen.device)
        # temp = idx.unsqueeze(0).expand(batchSize, maxLen)
        # curLen = resultLen.unsqueeze(1).expand(batchSize, maxLen)
        # mask = curLen >= temp
        # pr
        #
        # input = feature[idx, :resultLen, :]
        # sum = torch.sum(input, dim=(1))
        # self.out = self.act(sum.matmul(self.linear))
        out, hidden = self.Gru(feature, None)
        resultLen = resultLen.type(torch.int).to(device=resultLen.device)
        resultLen = (
            torch.where(resultLen == 0, torch.ones(resultLen.size()).type(torch.int).to(resultLen.device),
                        resultLen.type(torch.int).to(resultLen.device)))
        resultLen = resultLen - 1
        if resultLen.ndim != 2:
            resultLen = resultLen.squeeze().type(torch.long).to(resultLen.device)
            idx = torch.arange(len(resultLen), dtype=torch.long).to(resultLen.device)
        else:
            resultLen = resultLen.reshape([-1]).type(torch.long).to(resultLen.device)
            idx = torch.arange(len(resultLen), dtype=torch.long).to(resultLen.device)
        # print("idx", idx)
        self.out = out[idx, resultLen, :]
        out = self.linear(feature)
        outTemp = self.linear2(out.permute(0, 2, 1)).squeeze(2)
        return self.out
        # return outTemp
