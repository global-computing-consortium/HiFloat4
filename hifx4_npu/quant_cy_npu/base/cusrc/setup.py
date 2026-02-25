import os 
from setuptools import setup, Extension
from torch.utils import cpp_extension


# some global variables 
ascend_op_extension_name = 'npu_quant_op'
ascend_op_src_files = ['npu_quantop_base.cpp']
ascend_toolkit_install_path='/usr/local/Ascend/ascend-toolkit/latest'

pytorch_extension_name = 'npu_quant'
pytorch_interface_cpp_files = ['npu_quant.cpp']


# build ascend op
ascend_include_paths=[
    ascend_toolkit_install_path + '/aarch64-linux/tikcpp/tikcfw/',
    ascend_toolkit_install_path + '/aarch64-linux/tikcpp/tikcfw/interface',
    ascend_toolkit_install_path + '/aarch64-linux/tikcpp/tikcfw/impl',
    ascend_toolkit_install_path + '/aarch64-linux/tikcpp/tikcfw/kernel_tiling',
    ascend_toolkit_install_path + '/aarch64-linux/tikcpp/tikcfw/op_frame',
]
ascend_libraries = ['runtime', 'ascendcl']
ascend_library_paths = [
    ascend_toolkit_install_path + '/runtime/lib64',
]

cmd = 'ccec -O3 -xcce '
cmd += ' '.join(ascend_op_src_files) + ' '
cmd += '--cce-aicore-arch=dav-c220-vec '

cmd += ' '.join(['-L'+p for p in ascend_library_paths]) + ' '
cmd += ' '.join(['-I'+p for p in ascend_include_paths]) + ' '
cmd += ' '.join(['-l'+l for l in ascend_libraries]) + ' '
cmd += '-shared -fPIC -o lib' + ascend_op_extension_name + '.so -std=c++17 '

print(cmd)
os.system(cmd)


# build pytorch extension 
import torch 
import torch_npu
torch_npu_path = torch_npu.__file__
torch_npu_path = '/'.join(torch_npu_path.split('/')[:-1])
print('TORCH_NPU:', torch_npu_path)
extra_link_args = [
    f'-Wl,-rpath={torch_npu_path}/lib,-rpath={os.getcwd()}'
]
setup(name=pytorch_extension_name,
      ext_modules=[cpp_extension.CppExtension(pytorch_extension_name, pytorch_interface_cpp_files,
                                              include_dirs=[
                                                  torch_npu_path + '/include',
                                                  ],
                                              library_dirs=[
                                                  torch_npu_path + '/lib',
                                                  os.getcwd(),
                                                  ],
                                              libraries=[ascend_op_extension_name, 'torch_npu'],
                                              extra_link_args=extra_link_args,
                                            #   extra_compile_args=['-O2']
                                              )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
