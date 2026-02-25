from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from ..base.QType import QType
from ..base.QTensor import quant_dequant_float


class QuantFunc(Function):
    @staticmethod
    def forward(ctx, x: Tensor, Q:QType):
        ctx._Q = Q 
        if Q.desc=='bf16':
            return x.bfloat16()
        elif Q.desc=='fp16':
            return x.half()
        elif Q.desc=='fp32':
            return x.float()
        else:
            return quant_dequant_float(x, Q, force_py=False)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        Q = ctx._Q
        if Q.desc=='bf16':
            return grad_output
        elif Q.desc=='fp16':
            return grad_output
        elif Q.desc=='fp32':
            return grad_output
        else:
            return quant_dequant_float(grad_output, ctx._Q), None


class QuantFunc_keepgrad(Function):
    @staticmethod
    def forward(ctx, x: Tensor, Q:QType):
        ctx._Q = Q 
        if Q.desc=='bf16':
            return x.bfloat16()
        elif Q.desc=='fp16':
            return x.half()
        elif Q.desc=='fp32':
            return x.float()
        else:
            return quant_dequant_float(x, Q, force_py=False)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        Q = ctx._Q
        if Q.desc=='bf16':
            return grad_output, None
        elif Q.desc=='fp16':
            return grad_output, None
        elif Q.desc=='fp32':
            return grad_output, None
        else:
            return grad_output, None


class QuantFunc_keepinput(Function):
    @staticmethod
    def forward(ctx, x: Tensor, Q:QType):
        ctx._Q = Q 
        if Q.desc=='bf16':
            return x.bfloat16()
        elif Q.desc=='fp16':
            return x.half()
        elif Q.desc=='fp32':
            return x.float()
        else:
            return x
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        Q = ctx._Q
        if Q.desc=='bf16':
            return grad_output, None
        elif Q.desc=='fp16':
            return grad_output, None
        elif Q.desc=='fp32':
            return grad_output, None
        else:
            return quant_dequant_float(grad_output, ctx._Q), None


class QConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int,int], List[int]], \
                 stride: int, padding: int, dilation: int,\
                 groups: int, bias: bool=True):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.weight = Tensor(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        else:
            self.weight = Tensor(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.weight = nn.Parameter(self.weight)
        if bias:
            self.bias = Tensor(torch.randn(out_channels))
            self.bias = nn.Parameter(self.bias)
        else:
            self.bias = None 

        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation 
        self.groups = groups 

        self.qparams = None 
        self.in_qparams = None 
        self._quant_grad = True

    def set_quant_grad(self, value: bool):
        self._quant_grad = value

    def forward(self, x: Tensor):
        assert self.qparams is not None, 'Conv2d: Must assign quant params (QType) to this layer'

        qp = self.qparams
        qp_in = self.qparams if self.in_qparams is None else self.in_qparams

        if not x.is_contiguous():
            x = x.contiguous()

        x = QuantFunc_keepgrad.apply(x, qp_in)  # type: ignore
        w = QuantFunc_keepgrad.apply(self.weight, qp)  # type: ignore
        
        out = F.conv2d(x, w, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)  # type: ignore 

        if self._quant_grad:
            out = QuantFunc_keepinput.apply(out, qp_in)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1).unsqueeze(-1)
        return out 
    
    def transfer(self, layer: Union['QConv2d', nn.Conv2d]):
        self.to(layer.weight.device)
        self.to(layer.weight.dtype)
        self.weight.data[:] = layer.weight.data[:]
        if self.bias is not None:
            self.bias.data[:] = layer.bias.data[:]   # type: ignore

    def assign_qparams(self, Q: Union[QType, str]):
        if isinstance(Q, str):
            self.qparams = QType(Q).dim_(1)
        else:
            self.qparams = Q.copy().dim_(1)

    def assign_input_qparams(self, Q: Union[QType, str]):
        if isinstance(Q, str):
            self.in_qparams = QType(Q).dim_(1)
        else:
            self.in_qparams = Q.copy().dim_(1)

    def __deepcopy__(self, memo):
        layer = QConv2d(self.weight.shape[1], self.weight.shape[0], [self.weight.shape[2], self.weight.shape[3]], \
                        self.stride, self.padding, self.dilation, self.groups, self.bias is not None)
        layer.transfer(self)
        assert self.qparams is not None, 'Must assign quant params before deepcopy'
        layer.assign_qparams(self.qparams)
        if self.in_qparams is not None:
            layer.assign_input_qparams(self.in_qparams)
        return layer
    