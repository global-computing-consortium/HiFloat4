from typing import List, Union

import torch
import torch.nn as nn

from ..base.QType import QType
from ..layers.QConv import QConv2d
from ..layers.QLinear import QLinear
from ..layers.QSLinear import QSLinear
from ..layers.SLinear import SLinear


# layer conversion functions 
def replace_linear(module: nn.Module, w_Q: Union[QType, str], in_Q: Union[QType, str, None]=None, quant_grad: bool=True, exclude_layers: List[str]=[]):
    assert isinstance(exclude_layers, list), 'Exclude layers must be list of string'
    # record module names 
    mod_dict = {}
    for n,m in module.named_modules():
        mod_dict[n] = m 
    
    for n,m in module.named_modules():
        if n in exclude_layers:
            print('(Replace Qlinear) Excluding layer:', n)
            continue
        if isinstance(m, nn.Linear):
            new_mod = QLinear(m.in_features, m.out_features, m.bias is not None)
            new_mod.transfer(m)
            new_mod.assign_qparams(w_Q)
            new_mod.set_quant_grad(quant_grad)
            # if quant_qkv and (('q_proj' in n) or ('k_proj' in n) or ('v_proj' in n)):
            #     new_mod.set_quant_output(True)

            if in_Q is not None:
                new_mod.assign_input_qparams(in_Q)

            parent_mod = mod_dict['.'.join(n.split('.')[:-1])]
            setattr(parent_mod, n.split('.')[-1], new_mod)
            
# layer conversion functions 
def replace_sparse_quant_linear(module: nn.Module, w_Q: Union[QType, str], in_Q: Union[QType, str, None]=None, quant_grad=True, calibration_dict=None):
    # record module names 
    mod_dict = {}
    for n,m in module.named_modules():
        mod_dict[n] = m 
    
    for n,m in module.named_modules():
        if isinstance(m, nn.Linear):
            sparse_ratio_n = calibration_dict[n] if calibration_dict is not None else 0.0
            new_mod = QSLinear(m.in_features, m.out_features, m.bias is not None, sparse_ratio=sparse_ratio_n)
            new_mod.transfer(m)
            new_mod.assign_qparams(w_Q)
            new_mod.set_quant_grad(quant_grad)

            if in_Q is not None:
                new_mod.assign_input_qparams(in_Q)

            parent_mod = mod_dict['.'.join(n.split('.')[:-1])]
            setattr(parent_mod, n.split('.')[-1], new_mod)


# layer conversion functions 
def replace_sparse_linear(module: nn.Module, calibration_dict=None):
    # record module names 
    mod_dict = {}
    for n,m in module.named_modules():
        mod_dict[n] = m 
    
    for n,m in module.named_modules():
        if isinstance(m, nn.Linear):
            sparse_ratio_n = calibration_dict[n] if calibration_dict is not None else 0.0
            new_mod = SLinear(m.in_features, m.out_features, m.bias is not None, sparse_ratio=sparse_ratio_n)
            new_mod.transfer(m)

            parent_mod = mod_dict['.'.join(n.split('.')[:-1])]
            setattr(parent_mod, n.split('.')[-1], new_mod)



def replace_linear_mixfp(module: nn.Module, w_Q: Union[QType, str], high_Q: Union[QType, str], ratio: float=0.0, quant_grad=True):
    high_prec_layer_names = []
    if ratio>0:
        w_quant_desc = w_Q if isinstance(w_Q, str) else w_Q.desc
        quant_err_list = torch.load(f'mix_fp/mixfp_err_{w_quant_desc}.pt')
        n_layers = int(ratio * len(quant_err_list))
        high_prec_layer_names = [i[0] for i in quant_err_list[-n_layers:]]
        print(f'{n_layers} layers will be assigned to high precision bit: {high_Q}')

    # record module names 
    mod_dict = {}
    for n,m in module.named_modules():
        mod_dict[n] = m 
    
    for n,m in module.named_modules():
        if isinstance(m, nn.Linear):
            new_mod = QLinear(m.in_features, m.out_features, m.bias is not None)
            new_mod.transfer(m)
            if n in high_prec_layer_names:
                new_mod.assign_qparams(high_Q)
                # print(f'Layer {n} will be assigned to {high_Q}')
            else:
                new_mod.assign_qparams(w_Q)
            new_mod.set_quant_grad(quant_grad)

            parent_mod = mod_dict['.'.join(n.split('.')[:-1])]
            setattr(parent_mod, n.split('.')[-1], new_mod)


def replace_conv2d(module: nn.Module, w_Q: QType, in_Q: Union[QType, None]=None, quant_grad=True):
    # record module names 
    mod_dict = {}
    for n,m in module.named_modules():
        mod_dict[n] = m 
    
    for n,m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            new_mod = QConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, m.bias is not None)  # type: ignore
            new_mod.transfer(m)
            new_mod.assign_qparams(w_Q)
            if in_Q is not None:
                new_mod.assign_input_qparams(in_Q)
            new_mod.set_quant_grad(quant_grad)

            parent_mod = mod_dict['.'.join(n.split('.')[:-1])]
            setattr(parent_mod, n.split('.')[-1], new_mod)
        
    
def assign_qparams(module: nn.Module, w_Q: Union[QType, str], in_Q: Union[QType, str, None]=None):
    for n,m in module.named_modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.assign_qparams(w_Q)
            if in_Q is not None:
                m.assign_input_qparams(in_Q)


def set_fastforward(module: nn.Module, value: bool=True):
    print('Switch QLinear layers to fast_forward mode:', value)
    for n,m in module.named_modules():
        if isinstance(m, QLinear):
            m._fast_forward = value

