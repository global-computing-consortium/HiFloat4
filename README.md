# HiFloat4

HiFloat4: High-Performance HiFloat4 Quantization Library.

HiFloat4 is a library designed for efficient Float4 quantization and simulation across different hardware backends, including NVIDIA CUDA and Huawei Ascend NPU. It provides high-performance kernels for pseudo-quantization, enabling researchers to simulate HiFloat4 precision in deep learning models.

Installation & Verification
1. CUDA Version (NVIDIA GPUs)
To build and verify the CUDA-accelerated operators, follow these steps:

```bash
# Build the CUDA kernels
bash build.sh

# Run the verification script
python hif4.py

# Verification:
If the output: displays ABS diff max (zero values): 0
The installation is successful, and the results match the reference.
```
2. NPU Version (Huawei Ascend)
To build and verify the NPU-accelerated operators, follow these steps:
```bash
# Build the NPU kernels
bash build_npu_ops.sh

# Run the verification script
python hif4.py

# Verification:
If the output: displays ABS diff max (zero values): 0.
The installation is working correctly on the Ascend hardware.
```

3. Usage: Standard Linear Layer Simulation (GPU Example). To simulate a HiFloat4 Linear layer using pseudo-quantization on a GPU, you should quantize both the input $x$ and the weights $w$ before performing the standard linear operation. The standard workflow for the GPU platform is as follows:
```bash
import torch
from quant_cy import QType, quant_dequant_float #gpu

# 1. Prepare your input and weights
# 2. Apply quant-dequant simulation
# Note: Ensure tensors are on the correct device (e.g., .cuda())
qtype_str = 'hifx4'
print('Qtype string: %s '%(qtype_str))
quant_type = QType(qtype_str).dim(0) 
x_sim = quant_dequant_float(x.cuda(), quant_type, force_py=False, force_fp32=True)
w_sim = quant_dequant_float(w.cuda(), quant_type, force_py=False, force_fp32=True)

# 3. Execute the linear layer
y = torch.nn.functional.linear(x_sim, w_sim)
```

4. Usage: Standard Linear Layer Simulation (NPU Example):
```bash
import torch
from quant_cy_npu import QType, quant_dequant_float #npu

# 1. Prepare your input and weights
# 2. Apply quant-dequant simulation
# Note: Ensure tensors are on the correct device (e.g., .npu())
qtype_str = 'hifx4'
print('Qtype string: %s '%(qtype_str))
quant_type = QType(qtype_str).dim(0) 
x_sim = quant_dequant_float(x.npu(), quant_type, force_py=False, force_fp32=True)
w_sim = quant_dequant_float(w.npu(), quant_type, force_py=False, force_fp32=True)

# 3. Execute the linear layer
y = torch.nn.functional.linear(x_sim, w_sim)
```

# Citation
If you find this work useful for your research, please cite the following paper:
```bash
@misc{luo2026hifloat4formatlanguagemodel,
      title={HiFloat4 Format for Language Model Inference}, 
      author={Yuanyong Luo and Jing Huang and Yu Cheng and Ziwei Yu and Kaihua Tang and Xinda Ma and Xin Wang and Anping Tong and Guipeng Hu and Yun Xu and Mehran Taghian and Peng Wu and Guanglin Li and Yunke Peng and Tianchi Hu and Minqi Chen and Michael Bi Mi and Hu Liu and Xiping Zhou and Junsong Wang and Qiang Lin and Heng Liao},
      year={2026},
      eprint={2602.11287},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.11287}, 
}
```

