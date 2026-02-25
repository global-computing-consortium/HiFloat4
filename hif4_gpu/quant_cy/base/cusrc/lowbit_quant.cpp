#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void mxfp4_quant_cuda(const torch:: Tensor x, torch::Tensor result);
void mxfp4_quant_cuda_bf16(const torch:: Tensor x, torch::Tensor result);
void mxfp6_quant_cuda(const torch:: Tensor x, torch::Tensor result);
void mxfp6_quant_cuda_bf16(const torch:: Tensor x, torch::Tensor result);
void mxfp8e4m3_quant_cuda(const torch:: Tensor x, torch::Tensor result);
void mxfp8e4m3_quant_cuda_bf16(const torch:: Tensor x, torch::Tensor result);
void nvf4_quant_cuda(const torch:: Tensor x, torch::Tensor result);
void nvf4_quant_cuda_bf16(const torch:: Tensor x, torch::Tensor result);


void mxfp4_quant(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    mxfp4_quant_cuda(x, result);
}

void mxfp4_quant_bf16(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    mxfp4_quant_cuda_bf16(x, result);
}

void mxfp6_quant(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    mxfp6_quant_cuda(x, result);
}

void mxfp6_quant_bf16(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    mxfp6_quant_cuda_bf16(x, result);
}

void mxfp8e4m3_quant(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    mxfp8e4m3_quant_cuda(x, result);
}

void mxfp8e4m3_quant_bf16(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    mxfp8e4m3_quant_cuda_bf16(x, result);
}

void nvf4_quant(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    nvf4_quant_cuda(x, result);
}

void nvf4_quant_bf16(const torch::Tensor x, torch::Tensor result){
    CHECK_INPUT(x);
    CHECK_INPUT(result);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    nvf4_quant_cuda_bf16(x, result);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("mxfp4_quant", &mxfp4_quant, "mxfp4_quant", py::arg("x"), py::arg("result"));
    m.def("mxfp4_quant_bf16", &mxfp4_quant_bf16, "mxfp4_quant_bf16", py::arg("x"), py::arg("result"));
    m.def("mxfp6_quant", &mxfp6_quant, "mxfp6_quant", py::arg("x"), py::arg("result"));
    m.def("mxfp6_quant_bf16", &mxfp6_quant_bf16, "mxfp6_quant_bf16", py::arg("x"), py::arg("result"));
    m.def("mxfp8e4m3_quant", &mxfp8e4m3_quant, "mxfp8e4m3_quant", py::arg("x"), py::arg("result"));
    m.def("mxfp8e4m3_quant_bf16", &mxfp8e4m3_quant_bf16, "mxfp8e4m3_quant_bf16", py::arg("x"), py::arg("result"));
    m.def("nvf4_quant", &nvf4_quant, "nvf4_quant", py::arg("x"), py::arg("result"));
    m.def("nvf4_quant_bf16", &nvf4_quant_bf16, "nvf4_quant_bf16", py::arg("x"), py::arg("result"));
}

