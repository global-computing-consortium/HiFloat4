import torch 
from quant_cy import QType, quant_dequant_float
import numpy as np 
import HiF4_NVFP4_v14f16


np.random.seed(42)
torch.manual_seed(42)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)}, linewidth=65)


if __name__=='__main__':
    ##  生成随机变量
    M = 512
    N = 512
    x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32)
    x_torch = torch.from_numpy(x).cuda()
    print(x.shape)

    print('Testing zero values')
    y1 = HiF4_NVFP4_v14f16.To_HiFX(x*0, N=N)
    y2 = quant_dequant_float((x_torch*0), QType('hifx4').dim(0), force_py=False).float().cpu().numpy()
    print(y2.max(), y2.min())
    diff = np.abs(y1 - y2)
    print('ABS diff max (zero values):', np.max(diff))
