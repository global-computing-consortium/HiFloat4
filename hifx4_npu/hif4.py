import torch 
import torch_npu
from quant_cy_npu import QType, quant_dequant_float
import numpy as np 
import HiF4_NVFP4_v14f16
import time 

np.random.seed(42)

N = 1024
M = 1024
NN = N
MM = M 
# x = np.load('problem.npy')[...,None]
x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32)
x_torch = torch.from_numpy(x).npu()
print(x.shape)

for N in [4]:
    # pytorch 实现
    qtype_str = 'hifx%d'%(N)
    print('Testing N=%d  Qtype string: %s '%(N, qtype_str))
    quant_type = QType(qtype_str).dim(0)
    
    y0 = HiF4_NVFP4_v14f16.To_HiFX(x, N=N)

    torch.npu.synchronize()
    start = time.time()
    y1 = quant_dequant_float(x_torch, quant_type, force_py=True, force_fp32=False)
    torch.npu.synchronize()
    print('Time (force=True) :', time.time()-start)

    torch.npu.synchronize()
    start = time.time()
    y2 = quant_dequant_float(x_torch, quant_type, force_py=False, force_fp32=False)
    torch.npu.synchronize()
    print('Time (force=False) :', time.time()-start)

    y1 = y1.cpu().numpy()
    y2 = y2.cpu().numpy()

    print(y2)
    diff = np.abs(y0 - y1)
    print('ABS diff max (numpy <-> torch ):', np.max(diff))
    diff = np.abs(y0 - y2)
    print('ABS diff max (numpy <-> kernel):', np.max(diff))
    diff = np.abs(y1 - y2)
    print('ABS diff max (torch <-> kernel):', np.max(diff))
    

print('Testing zero values')
y1 = HiF4_NVFP4_v14f16.To_HiFX(x*0, N=N)
y2 = quant_dequant_float((x_torch*0).npu(), QType('hifx4').dim(0), force_py=False).cpu().numpy()
diff = np.abs(y1 - y2)
print('ABS diff max (zero values):', np.max(diff))
