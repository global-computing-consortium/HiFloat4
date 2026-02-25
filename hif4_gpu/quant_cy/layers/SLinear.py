import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor
from torch.autograd import Function


class SLinear(nn.Module):
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


    def forward(self, x: Tensor):
        # apply linear sparse
        if self.sparse_ratio > 0:
            x_dim = x.shape[-1]
            sparse_num = int(x_dim * self.sparse_ratio)
            assert sparse_num > 0, f"x dim is {x_dim}, sparse ratio is {self.sparse_ratioio}"
            thres = torch.kthvalue(x.abs().float(), k=sparse_num, dim=-1)[0].to(x.dtype)
            x[x.abs()<=thres.unsqueeze(-1)] = 0.0
        out = F.linear(x, self.weight, self.bias)
        return out 
    
    def transfer(self, layer: nn.Linear):
        self.to(layer.weight.device)
        self.to(layer.weight.dtype)
        self.weight.data[:] = layer.weight.data[:]
        if self.bias is not None:
            self.bias.data[:] = layer.bias.data[:]  # type: ignore