import torch 
from ..QType import QType
from torch import Tensor 


def quant_int8sym(x: Tensor, Q: QType, qdim: int):
    xmax = torch.abs(x).max(dim=qdim, keepdim=True)[0]
    interval = xmax / 127
    quanted = x / interval
    quanted = torch.round(quanted).clip(min=-127, max=127)
    recovered = quanted * interval 
    return recovered
