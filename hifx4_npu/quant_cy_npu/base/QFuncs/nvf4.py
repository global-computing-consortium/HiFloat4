import torch 
from ..QType import QType 
from torch import Tensor 


@torch.no_grad()
def quant_nvf4(x: Tensor, Q: QType, qdim: int): 
    x = x.unflatten(qdim, (-1, 16))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    grp_max = torch.amax(x_unsigned, dim=qdim, keepdim=True)
    sf = grp_max / 6
    sf = torch.clip_(sf, 0, 448)
    if x.dtype==torch.float16:
        sf_exp = torch.floor(torch.log2(sf + 2**-14))
    else:
        sf_exp = torch.floor(torch.log2(sf + 2**-60))
    sf_exp.clamp_(-6)
    E4M3 = torch.round(sf * 2 ** (-sf_exp + 3)) * 2 ** (-3 + sf_exp)

    if x.dtype==torch.float16:
        E4M3 = E4M3 + 2**(-14)
    else:
        E4M3 = E4M3 + 2**(-60)
    
    igv = x_unsigned / E4M3
    if x.dtype==torch.float16:
        E2 = torch.floor(torch.log2(igv + 2**(-14)))
    else:
        E2 = torch.floor(torch.log2(igv + 2**(-60)))
    E2.clamp_(0)
    M1 = torch.round(igv*2**(-E2 + 1))*2**(-1)
    E2M1 = 2**E2*M1
    E2M1.clamp_(0, 6)

    res = sign * E4M3 * E2M1

    # E2M1 = torch.ones_like(x)
    # res = E4M3 * E2M1
    
    res = res.flatten(qdim-1, qdim)
    return res 
