import torch
import torch.nn as nn
from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.BaseTransforms.L0Regularization import L0Regularization
from Models.BaseTransforms.Attention import DotWeight


# [B,F,D] *[F,D]->[B,F,D]
class FeaturePruningLayer(nn.Module):
    def __init__(self, featureNumb, inDim, outDim, beta, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.outDim = outDim
        self.inDim = inDim
        self.zeta = zeta
        self.beta = beta
        self.gamma = gamma
        self.linear = nn.Linear(inDim, outDim, bias=True)
        self.act = nn.ReLU(inplace=False)
        # with torch.no_grad():
        #     self.linear.weight.copy_(self.linear.weight + 0.2)
        self.L0Reg = L0Regularization(beta, zeta, gamma)
        self.L0Out = None
        self.L0 = None
        self.output = None

    def forward(self, feature: torch.Tensor, indicator: torch.Tensor):
        linearOut = self.act(self.linear(indicator))
        L0Out, L0 = self.L0Reg(linearOut)
        self.L0 = L0
        pruning = feature * L0Out[None, :, :]
        output = pruning
        return output, L0


##### element-wise product
class FeaturePruningLayerV2(nn.Module):
    def __init__(self, featureNumb, inDim, outDim, beta, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.outDim = outDim
        self.inDim = inDim
        self.zeta = zeta
        self.beta = beta
        self.gamma = gamma
        self.linear = nn.Linear(inDim, outDim, bias=True)
        self.act = nn.ReLU(inplace=False)
        # with torch.no_grad():
        #     self.linear.weight.copy_(self.linear.weight + 0.2)
        self.L0Reg = L0Regularization(beta, zeta, gamma)
        self.L0Out = None
        self.L0 = None
        self.output = None

    # feature:[B,F,D] *[1,F,D] => [B,F,D]
    def forward(self, feature: torch.Tensor, indicator: torch.Tensor):
        indicator = feature * indicator[None, :, :]
        linearOut = self.act(self.linear(indicator))
        L0Out, L0 = self.L0Reg(linearOut)
        self.L0 = L0
        pruning = feature * L0Out
        output = pruning
        return output, L0


class StructPruningLayer(nn.Module):
    def __init__(self, inDim, outDim, beta, zeta=1.1, gamma=-0.1):
        super().__init__()
        self.outDim = outDim
        self.inDim = inDim
        self.linear = LinearTransform([inDim, outDim], dropLast=False)
        self.similarity = DotWeight()
        self.L0Reg = L0Regularization(beta, zeta, gamma)
        self.L0Out = None
        self.L0 = None
        self.output = None

    def forward(self, indicator: torch.Tensor):
        featureNumb = indicator.shape[-2]
        mask = torch.triu(torch.ones(featureNumb, featureNumb, device=indicator.device, requires_grad=False),
                          1)[:, :]
        linearOut = self.linear(indicator)
        score = self.similarity(linearOut, linearOut)
        score, _ = self.L0Reg(score)
        self.L0Out = torch.mul(mask, score)
        L0 = self.L0Reg.calL0(score)
        maskL0 = torch.mul(mask, L0)
        self.L0 = torch.mean(maskL0)
        return self.L0Out

# if __name__ == '__main__':
#     layer_count = 4
#
#     model = nn.LSTM(10, 20, num_layers=layer_count, bidirectional=True)
#     model.eval()
#
#     with torch.no_grad():
#         input = torch.randn(5, 3, 10)
#         h0 = torch.randn(layer_count * 2, 3, 20)
#         c0 = torch.randn(layer_count * 2, 3, 20)
#         output, (hn, cn) = model(input, (h0, c0))
#
#         # default export
#         torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')
#         onnx_model = onnx.load('lstm.onnx')
#         # input shape [5, 3, 10]
#         print(onnx_model.graph.input[0])
#
#         # export with `dynamic_axes`
#         torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx',
#                           input_names=['input', 'h0', 'c0'],
#                           output_names=['output', 'hn', 'cn'],
#                           dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})
#         onnx_model = onnx.load('lstm.onnx')
#         # input shape ['sequence', 3, 10]
#         print(onnx_model.graph.input[0])
