from typing import Union, List, Optional

import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd   # type: ignore

from ..base.QTensor import quant_dequant_float
from ..base.QType import QType


class LinearForward(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, w, b, qp, qp_in, quant_grad):
        x_q = quant_dequant_float(x, qp_in, force_fp32=True)
        w_q = quant_dequant_float(w, qp, force_fp32=True)
        ctx.save_for_backward(x, w)
        ctx.qp = qp 
        ctx.qp_in = qp_in
        ctx.quant_grad = quant_grad
        ctx.has_bias = b is not None
        out = F.linear(x_q, w_q, b)
        return out 
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        qp = ctx.qp
        qp_in = ctx.qp_in
        quant_grad = ctx.quant_grad
        x, w = ctx.saved_tensors
        x_q = quant_dequant_float(x, qp_in, force_fp32=True)
        w_q = quant_dequant_float(w, qp, force_fp32=True)

        if qp.desc=='hif4':
            hif4_max = 7680 / 8
            scale = grad_out.max() / hif4_max
            grad_out = grad_out / scale

        grad_out_quant = quant_dequant_float(grad_out, qp_in, force_fp32=True) if quant_grad else grad_out  # [B, L, Cout]
        grad_out_quant_trans = quant_dequant_float(grad_out.flatten(0,-2).transpose(-1,-2).contiguous(), qp_in, force_fp32=True) if quant_grad else grad_out.flatten(0,-2).transpose(-1,-2)

        if qp.desc=='hif4':
            grad_out_quant = grad_out_quant * scale

        grad_in = grad_out_quant @ w_q
        grad_w = grad_out_quant_trans @ x_q.flatten(0,-2)
        grad_b = grad_out.flatten(0,-2).sum(0) if ctx.has_bias else None 
        return grad_in, grad_w, grad_b, None, None, None
  

class LinearForwardFast(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, w, b, qp, qp_in, quant_grad):
        x_q = quant_dequant_float(x, qp_in, force_fp32=True)
        w_q = quant_dequant_float(w, qp, force_fp32=True)
        ctx.save_for_backward(x_q, w_q)
        ctx.qp = qp 
        ctx.qp_in = qp_in
        ctx.quant_grad = quant_grad
        ctx.has_bias = b is not None
        out = F.linear(x_q, w_q, b)
        return out 
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        qp = ctx.qp
        qp_in = ctx.qp_in
        quant_grad = ctx.quant_grad
        x_q, w_q = ctx.saved_tensors

        if qp.desc=='hif4':
            hif4_max = 7680 / 8
            scale = grad_out.max() / hif4_max
            grad_out = grad_out / scale

        grad_out_quant = quant_dequant_float(grad_out, qp_in, force_fp32=True) if quant_grad else grad_out  # [B, L, Cout]
        
        if qp.desc=='hif4':
            grad_out_quant = grad_out_quant * scale
        
        grad_in = grad_out_quant @ w_q
        grad_w = grad_out_quant.flatten(0,-2).transpose(-1,-2) @ x_q.flatten(0,-2)
        grad_b = grad_out.flatten(0,-2).sum(0) if ctx.has_bias else None 
        return grad_in, grad_w, grad_b, None, None, None


class QLinear(nn.Linear):
    _recorded_input: Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qparams = None 
        self.in_qparams = None 
        self._quant_grad = True
        self._fast_forward = False
        self._quant_output = False

    def set_quant_grad(self, value: bool):
        self._quant_grad = value

    def set_quant_output(self, value: bool):
        self._quant_output = value

    def forward(self, x: Tensor):
        assert self.qparams is not None, 'Linear: Must assign quant params (QType) to this layer'
        qp = self.qparams.dim(-1)
        qp_in = self.qparams.dim(-1) if self.in_qparams is None else self.in_qparams.dim(-1)

        if not x.is_contiguous():
            x = x.contiguous()

        if self._fast_forward:
            out: Optional[Tensor] = LinearForwardFast.apply(x, self.weight, self.bias, qp, qp_in, self._quant_grad)
        else:
            out: Optional[Tensor] = LinearForward.apply(x, self.weight, self.bias, qp, qp_in, self._quant_grad)

        assert out is not None, 'Error: Output is None in QLinear'

        if self._quant_output:
            out = quant_dequant_float(out, qp_in, force_fp32=True)
        return out 
    
    def transfer(self, layer: Union['QLinear', nn.Linear]):
        self.to(layer.weight.device)
        self.to(layer.weight.dtype)
        self.weight.data.view(-1)[:] = layer.weight.data.view(-1)[:]
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
        layer = QLinear(self.weight.shape[1], self.weight.shape[0], self.bias is not None)
        layer.transfer(self)
        assert self.qparams is not None, 'Must assign quant params before deepcopy'
        layer.assign_qparams(self.qparams)
        if self.in_qparams is not None:
            layer.assign_input_qparams(self.in_qparams)
        return layer

    def finialize_calibration(self, calib_type: str):
        assert self.qparams is not None, 'QParam should not be None in QLinear'

        if calib_type=='gptq':
            from ptq.gptq.gptq import GPTQ
            quantizer = GPTQ(self)
            self._recorded_input = self._recorded_input.to(self.weight.device)
            quantizer.add_batch(self._recorded_input, self._recorded_input)
            w2 = quantizer.fasterquant(qt=self.qparams, groupsize=64, blocksize=64, percdamp=0.1)
            self.weight.data[:] = w2.data[:]
            del w2 
            del self._recorded_input
            quantizer.free()



