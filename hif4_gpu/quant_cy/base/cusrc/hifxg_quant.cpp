#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void hifx_quant_cuda(const torch::Tensor x, torch::Tensor result, int N);
void hifx_quant_cuda_bf16(const torch::Tensor x, torch::Tensor result, int N);

void hifx_quant(const torch::Tensor x, torch::Tensor result, int N){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    hifx_quant_cuda(x, result, N);
}

void hifx_quant_bf16(const torch::Tensor x, torch::Tensor result, int N){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    hifx_quant_cuda_bf16(x, result, N);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("hifx_quant", &hifx_quant, "hifx_quant", py::arg("x"), py::arg("result"), py::arg("N"));
    m.def("hifx_quant_bf16", &hifx_quant_bf16, "hifx_quant_bf16", py::arg("x"), py::arg("result"), py::arg("N"));
}

