import torch
import torch.nn as nn
from torch.autograd import gradcheck


# DYNAMIC SPARSE TRAINING: FIND EFFICIENT SPARSE NETWORK FROM SCRATCH WITH TRAINABLE MASKED LAYERS


class LTETransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        one = torch.ones_like(input_)
        zero = torch.zeros_like(input_)
        re = torch.where(input_ > 0, one, zero)
        return re

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_input = grad_output.clone()
        input = torch.abs(input)
        temp = 2. - (4. * input)
        zero = torch.zeros_like(grad_output)
        print(111111111)
        out = torch.where(input <= 0.4, temp, zero)
        clause1 = input <= 1.0
        clause2 = input > 0.4
        out[clause2 & clause1] = 0.4
        # print(out)
        # print(grad_output)
        outPut = grad_output * out
        return outPut


