from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor
from torch.autograd import Function

from ..base.QTensor import quant_dequant_float
from ..base.QType import QType


class LinearForward(Function):
    @staticmethod
    def forward(ctx, x, w, b, qp, qp_in, ratio):
        # apply quant & dequant
        x_q = quant_dequant_float(x, qp_in)
        w_q = quant_dequant_float(w, qp)
        ctx.save_for_backward(x_q, w_q)
        ctx.qp = qp 
        ctx.qp_in = qp_in
        ctx.has_bias = b is not None
        # apply linear sparse
        if ratio > 0:
            x_dim = x_q.shape[-1]
            sparse_num = int(x_dim * ratio)
            assert sparse_num > 0, f"x dim is {x_dim}, sparse ratio is {ratio}"
            thres = torch.kthvalue(x_q.abs().float(), k=sparse_num, dim=-1)[0].to(x_q.dtype)
            x_q[x_q.abs()<=thres.unsqueeze(-1)] = 0.0
        out = F.linear(x_q, w_q, b)
        return out 
        
    @staticmethod
    def backward(ctx, grad_out):
        qp = ctx.qp
        qp_in = ctx.qp_in
        x_q, w_q = ctx.saved_tensors
        grad_out_quant = quant_dequant_float(grad_out, qp_in)  # [B, L, Cout]
        grad_in = grad_out_quant @ w_q
        grad_w = grad_out_quant.flatten(0,-2).transpose(-1,-2) @ x_q.flatten(0,-2)
        grad_b = grad_out.flatten(0,-2).sum(0) if ctx.has_bias else None 
        return grad_in, grad_w, grad_b, None, None


class QSLinear(nn.Module):
    inputs: List[Tensor]
    outputs: List[Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool=True, sparse_ratio: float=0):
        super().__init__()
        self.weight = Tensor(torch.randn(out_features, in_features))   
        self.weight = nn.Parameter(self.weight)
        if bias:
            self.bias = Tensor(torch.randn(out_features))
            self.bias = nn.Parameter(self.bias)
        else:
            self.bias = None 

        # set sparse_ratio
        self.sparse_ratio = sparse_ratio

        self.qparams = None 
        self.in_qparams = None 
        self._quant_grad = True

    def set_quant_grad(self, value: bool):
        self._quant_grad = value

    def forward(self, x: Tensor):
        assert self.qparams is not None, 'Linear: Must assign quant params (QType) to this layer'
        qp = self.qparams.dim(-1)
        qp_in = self.qparams.dim(-1) if self.in_qparams is None else self.in_qparams.dim(-1)

        if not x.is_contiguous():
            x = x.contiguous()
        # x = QuantFunc_keepgrad.apply(x, qp_in)
        # w = QuantFunc_keepgrad.apply(self.weight, qp)
        # out = F.linear(x, w)

        # if self._quant_grad:
        #     out = QuantFunc_keepinput.apply(out, qp_in)
        # out = out if self.bias is None else out + self.bias 
        out = LinearForward.apply(x, self.weight, self.bias, qp, qp_in, self.sparse_ratio)
        return out 
    
    def transfer(self, layer: Union['QSLinear', nn.Linear]):
        self.to(layer.weight.device)
        self.to(layer.weight.dtype)
        self.weight.data[:] = layer.weight.data[:]
        if self.bias is not None:
            self.bias.data[:] = layer.bias.data[:]  # type: ignore
    
    def assign_qparams(self, Q: Union[QType, str]):
        if isinstance(Q, str):
            self.qparams = QType(Q)
        else:
            self.qparams = Q.copy()

    def assign_input_qparams(self, Q: Union[QType, str]):
        if isinstance(Q, str):
            self.in_qparams = QType(Q)
        else:
            self.in_qparams = Q.copy()

    def __deepcopy__(self, memo):
        layer = QSLinear(self.weight.shape[1], self.weight.shape[0], self.bias is not None)
        layer.transfer(self)
        assert self.qparams is not None, 'Must assign quant params before deepcopy'
        layer.assign_qparams(self.qparams)
        if self.in_qparams is not None:
            layer.assign_input_qparams(self.in_qparams)
        return layer
    
