#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/Atomic.cuh>


using bf16 = at::BFloat16;
using athalf = at::Half;


__device__ __forceinline__ int round_rshift(int mant, int rshift){
    int m = (mant >> rshift);
    m = m + (m & 1);
    m = m << rshift;
    return m;
}


__device__ __forceinline__ void split_two_fp16(float& src, float& target1, float& target2){
    unsigned int tmp = *(unsigned int*) &src;
    unsigned int tmp2 = (tmp & 0xFFFF) << 16;
    tmp = tmp & 0xFFFF0000;
    target1 = *(float*) &tmp;
    target2 = *(float*) &tmp2;
}


__device__ __forceinline__ void combine_two_fp16(float& src1, float& src2, float& target){
    unsigned int tmp1 = *(unsigned int*) &src1;
    unsigned int tmp2 = *(unsigned int*) &src2;
    tmp2 = tmp2 >> 16;
    tmp1 = tmp1 + tmp2;
    target = *(float*) &tmp1;
}


__device__ __forceinline__ void load_to_shared(const bf16* x_offset, float* x_shared, int& offset, int idx, const int& n_total){
    if ((offset + idx * 8)>=n_total){
        ((float4*) x_shared)[idx*2] = make_float4(0,0,0,0);
        ((float4*) x_shared)[idx*2+1] = make_float4(0,0,0,0);
    }else{
        float4 tmp = ((float4*)x_offset)[idx];
        split_two_fp16(tmp.x, x_shared[idx*8], x_shared[idx*8+1]);
        split_two_fp16(tmp.y, x_shared[idx*8+2], x_shared[idx*8+3]);
        split_two_fp16(tmp.z, x_shared[idx*8+4], x_shared[idx*8+5]);
        split_two_fp16(tmp.w, x_shared[idx*8+6], x_shared[idx*8+7]);
    }
}


__device__ __forceinline__ void store_to_shared(bf16* res_mem, float* res_shared, int& offset, int idx, const int& n_total){
    if ((offset + idx * 8) < n_total){
        combine_two_fp16(res_shared[idx*8+0], res_shared[idx*8+1], ((float*) res_mem)[idx*4]);
        combine_two_fp16(res_shared[idx*8+2], res_shared[idx*8+3], ((float*) res_mem)[idx*4+1]);
        combine_two_fp16(res_shared[idx*8+4], res_shared[idx*8+5], ((float*) res_mem)[idx*4+2]);
        combine_two_fp16(res_shared[idx*8+6], res_shared[idx*8+7], ((float*) res_mem)[idx*4+3]);
    }
}


__device__ void mxfp4_quant_cuda_inner(float* x_shared, float* res_shared, const int& thread_idx){
    float* x = x_shared + 32 * thread_idx;
    float* res = res_shared + 32 * thread_idx;

    // use offset to avoid bank conflict 
    int bank_offset = thread_idx % 32;

    // find shared exp 
    int exp_block = 0;
    for (int ii=0;ii<32;++ii){
        int i = (ii+bank_offset)%32;
        int xi = * (int*) (x+i);
        int exp = xi & 0x7F800000;
        if (exp > exp_block){ exp_block = exp; }
    }

    for (int ii=0;ii<32;++ii){
        int i = (ii+bank_offset)%32;
        if (*(x+i) == 0){
            res[i] = 0;
        }else{
            int xi = * (int*) (x+i);
            int mant = 0x00800000 + (xi & 0x00600000);
            
            // constrain exp within exp_block-2
            int e2 = (xi & 0x7F800000);
            int e3 = e2;
            if ((exp_block - e2) > 0x01000000){ e3 = exp_block - 0x01000000;}

            // round mantissa 
            int rshift = 21 + ((e3-e2)>>23);
            int m_ = (mant >> rshift);
            m_ = (m_ + (m_ & 1)) >> 1;

            // clip mantissa if value overflow 
            if (e3==exp_block && m_>3){ m_= 3; }

            // combine to float number 
            int expdiff = (exp_block - e3) >> 23;
            float mant_buff = ((float)m_) * 2.0 / ((float)(1 << expdiff));
            int resi = (xi & 0x80000000) | (exp_block - 0x01000000) ; 
            res[i] = (*(float*) &resi) * mant_buff;
        }
    }
}


__global__ void mxfp4_quant_cuda_kernel(const float* x_ori, float* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 32 * 128;
    const float* x_offset = x_ori + offset;
    float* res_mem = res_ori + offset;

    __shared__ float x_shared[32*128];
    __shared__ float res_shared[32*128];

    for (int i=0;i<8;++i){
        if ((offset + (i*block_size + thread_idx)*4) >= n_total){
            ((float4*) x_shared)[i*block_size+thread_idx] = make_float4(0,0,0,0);
        }else{
            ((float4*) x_shared)[i*block_size+thread_idx] = ((float4*)x_offset)[i*block_size+thread_idx];
        }
    }
    __syncthreads();
    
    mxfp4_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<8;++i){
        if ((offset + (i*block_size + thread_idx)*4) < n_total){
            ((float4*)res_mem)[i*block_size+thread_idx] = ((float4*)res_shared)[i*block_size+thread_idx];
        }
    }
}


__global__ void mxfp4_quant_cuda_bf16_kernel(const bf16* x_ori, bf16* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 32 * 128;
    const bf16* x_offset = x_ori + offset;
    bf16* res_mem = res_ori + offset;

    __shared__ float x_shared[32*128];
    __shared__ float res_shared[32*128];

    for (int i=0;i<4;++i){
        load_to_shared(x_offset, x_shared, offset, i* block_size+thread_idx, n_total);
    }
    __syncthreads();
    
    mxfp4_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<4;++i){
        store_to_shared(res_mem, res_shared, offset, i* block_size+thread_idx, n_total);
    }
}


void mxfp4_quant_cuda(const torch::Tensor x, torch::Tensor result){
    int threads = 128;
    int blocks = (x.numel() + threads*32 - 1) / (threads * 32);

    mxfp4_quant_cuda_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        result.data_ptr<float>(),
        x.numel()
    );
}


void mxfp4_quant_cuda_bf16(const torch::Tensor x, torch::Tensor result){
    int threads = 128;
    int blocks = (x.numel() + threads*32 - 1) / (threads * 32);

    mxfp4_quant_cuda_bf16_kernel<<<blocks, threads>>>(
        x.data_ptr<bf16>(),
        result.data_ptr<bf16>(),
        x.numel()
    );
}




__device__ __forceinline__ void mxfp6_quant_cuda_inner(float* x_shared, float* res_shared, const int& thread_idx){
    float* x = x_shared + 32 * thread_idx;
    float* res = res_shared + 32 * thread_idx;

    // use offset to avoid bank conflict 
    int bank_offset = thread_idx % 32;

    // find shared exp 
    int exp_block = 0;
    for (int ii=0;ii<32;++ii){
        int i = (ii+bank_offset)%32;
        int xi = * (int*) (x+i);
        int exp = xi & 0x7F800000;
        if (exp > exp_block){ exp_block = exp; }
    }

    for (int ii=0;ii<32;++ii){
        int i = (ii+bank_offset)%32;
        if (*(x+i) == 0){
            res[i] = 0;
        }else{
            int xi = * (int*) (x+i);
            int mant = 0x00800000 + (xi & 0x00780000);
            
            // constrain exp within exp_block-2
            int e2 = (xi & 0x7F800000);
            int e3 = e2;
            if ((exp_block - e2) > 0x01000000){ e3 = exp_block - 0x01000000;}

            // round mantissa 
            int rshift = 19 + ((e3-e2)>>23);
            int m_ = (mant >> rshift);
            m_ = (m_ + (m_ & 1)) >> 1;

            // clip mantissa if value overflow 
            if (e3==exp_block && m_>15){ m_= 15; }

            // combine to float number 
            int expdiff = (exp_block - e3) >> 23;
            float mant_buff = ((float)m_) / 2.0 / ((float)(1 << expdiff));
            int resi = (xi & 0x80000000) | (exp_block - 0x01000000) ; 
            res[i] = (*(float*) &resi) * mant_buff;
        }
    }
}


__global__ void mxfp6_quant_cuda_kernel(const float* x_ori, float* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 32 * 128;
    const float* x_offset = x_ori + offset;
    float* res_mem = res_ori + offset;

    __shared__ float x_shared[32*128];
    __shared__ float res_shared[32*128];

    for (int i=0;i<8;++i){
        if ((offset + i*block_size + thread_idx) >= n_total){
            ((float4*) x_shared)[i*block_size+thread_idx] = make_float4(0,0,0,0);
        }else{
            ((float4*) x_shared)[i*block_size+thread_idx] = ((float4*)x_offset)[i*block_size+thread_idx];
        }
    }
    __syncthreads();
    
    mxfp6_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<8;++i){
        if ((offset + i*block_size + thread_idx) < n_total){
            ((float4*)res_mem)[i*block_size+thread_idx] = ((float4*)res_shared)[i*block_size+thread_idx];
        }
    }
}


__global__ void mxfp6_quant_cuda_bf16_kernel(const bf16* x_ori, bf16* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 32 * 128;
    const bf16* x_offset = x_ori + offset;
    bf16* res_mem = res_ori + offset;

    __shared__ float x_shared[32*128];
    __shared__ float res_shared[32*128];

    for (int i=0;i<4;++i){
        load_to_shared(x_offset, x_shared, offset, i*block_size+thread_idx, n_total);
    }
    __syncthreads();
    
    mxfp6_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<4;++i){
        store_to_shared(res_mem, res_shared, offset, i*block_size+thread_idx, n_total);
    }
}


void mxfp6_quant_cuda(const torch::Tensor x, torch::Tensor result){
    int threads = 128;
    int blocks = (x.numel() + 128*32 - 1) / (128 * 32);
    
    mxfp6_quant_cuda_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        result.data_ptr<float>(),
        x.numel()
    );
}


void mxfp6_quant_cuda_bf16(const torch::Tensor x, torch::Tensor result){
    int threads = 128;
    int blocks = (x.numel() + 128*32 - 1) / (128 * 32);
    
    mxfp6_quant_cuda_bf16_kernel<<<blocks, threads>>>(
        x.data_ptr<bf16>(),
        result.data_ptr<bf16>(),
        x.numel()
    );
}





__device__ __forceinline__ void mxfp8e4m3_quant_cuda_inner(float* x_shared, float* res_shared, const int& thread_idx){
    float* x = x_shared + 32 * thread_idx;
    float* res = res_shared + 32 * thread_idx;

    // use offset to avoid bank conflict 
    int bank_offset = thread_idx % 32;

    // find shared exp 
    int exp_block = 0;
    for (int ii=0;ii<32;++ii){
        int i = (ii+bank_offset)%32;
        int xi = * (int*) (x+i);
        int exp = xi & 0x7F800000;
        if (exp > exp_block){ exp_block = exp; }
    }

    for (int ii=0;ii<32;++ii){
        int i = (ii+bank_offset)%32;
        if (*(x+i) == 0){
            res[i] = 0;
        }else{
            int xi = * (int*) (x+i);
            int mant = 0x00800000 + (xi & 0x00780000);
            
            // constrain exp within exp_block-14
            int e2 = (xi & 0x7F800000);
            int e3 = e2;
            if ((exp_block - e2) > 0x07000000){ e3 = exp_block - 0x07000000;}

            // round mantissa 
            int rshift = 19 + ((e3-e2)>>23);
            int m_ = (mant >> rshift);
            m_ = (m_ + 1) >> 1;

            // clip mantissa if value overflow 
            if (e3==exp_block && m_>14){ m_= 14; }

            // combine to float number 
            int expdiff = (exp_block - e3) >> 23;
            float mant_buff = ((float)m_) / 2.0 / ((float)(1 << expdiff));
            int resi = (xi & 0x80000000) | (exp_block - 0x01000000) ; 
            res[i] = (*(float*) &resi) * mant_buff;
        }
    }
}


__global__ void mxfp8e4m3_quant_cuda_kernel(const float* x_ori, float* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 32 * 128;
    const float* x_offset = x_ori + offset;
    float* res_mem = res_ori + offset;

    __shared__ float x_shared[32*128];
    __shared__ float res_shared[32*128];

    for (int i=0;i<8;++i){
        if ((offset + i*block_size + thread_idx) >= n_total){
            ((float4*) x_shared)[i*block_size+thread_idx] = make_float4(0,0,0,0);
        }else{
            ((float4*) x_shared)[i*block_size+thread_idx] = ((float4*)x_offset)[i*block_size+thread_idx];
        }
    }
    __syncthreads();
    
    mxfp8e4m3_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<8;++i){
        if ((offset + i*block_size + thread_idx) < n_total){
            ((float4*)res_mem)[i*block_size+thread_idx] = ((float4*)res_shared)[i*block_size+thread_idx];
        }
    }
}


__global__ void mxfp8e4m3_quant_cuda_bf16_kernel(const bf16* x_ori, bf16* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 32 * 128;
    const bf16* x_offset = x_ori + offset;
    bf16* res_mem = res_ori + offset;

    __shared__ float x_shared[32*128];
    __shared__ float res_shared[32*128];

    for (int i=0;i<4;++i){
        load_to_shared(x_offset, x_shared, offset, i*block_size+thread_idx, n_total);
    }
    __syncthreads();
    
    mxfp8e4m3_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<4;++i){
        store_to_shared(res_mem, res_shared, offset, i*block_size+thread_idx, n_total);
    }
}


void mxfp8e4m3_quant_cuda(const torch::Tensor x, torch::Tensor result){
    int threads = 128;
    int blocks = (x.numel() + 128*32 - 1) / (128 * 32);
    
    mxfp8e4m3_quant_cuda_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        result.data_ptr<float>(),
        x.numel()
    );
}


void mxfp8e4m3_quant_cuda_bf16(const torch::Tensor x, torch::Tensor result){
    int threads = 128;
    int blocks = (x.numel() + 128*32 - 1) / (128 * 32);
    
    mxfp8e4m3_quant_cuda_bf16_kernel<<<blocks, threads>>>(
        x.data_ptr<bf16>(),
        result.data_ptr<bf16>(),
        x.numel()
    );
}




__device__ __forceinline__ void nvf4_quant_cuda_inner(float* x_shared, float* res_shared, const int& thread_idx){
    float* x = x_shared + 16 * thread_idx;
    float* res = res_shared + 16 * thread_idx;

    // use offset to avoid bank conflict 
    int bank_offset = thread_idx % 16;

    // find max value 
    float max_val = 0;
    for (int i=0; i<16; ++i){
        if (abs(x[i]) > max_val){
            max_val = abs(x[i]);
        }
    }

    if (max_val==0){
        for (int i=0; i<16; ++i){
            res[i] = 0;
        }
        return;
    }

    // group scale = e4m3(maxval/6)
    max_val = max_val / 6.0f;
    if (max_val > 448){ max_val = 448; }
    uint scale_exp_int = *(uint*) &max_val;
    scale_exp_int = scale_exp_int & 0x7F800000;
    if (scale_exp_int < 0x3C800000){ scale_exp_int = 0x3C800000; }
    float scale_exp = *(float*) &scale_exp_int;
    // get mant 
    float mant = round(max_val * 8 / scale_exp);
    float scale = mant / 8 * scale_exp;
    if (scale==0){
        for (int i=0; i<16; ++i){
            res[i] = 0;
        }
        return;
    }

    // get e2m1 for each element
    for (int i=0; i<16; ++i){
        if (x[i]==0){res[i]=0; continue;}
        float v = x[i] / scale;
        uint v_int = *(uint*) &v;
        uint v_exp_int = v_int & 0x7F800000;
        if (v_exp_int<0x3F800000){ v_exp_int = 0x3F800000; }
        float v_exp = *(float*) &v_exp_int;

        float m1 = round(abs(v) * 2 / v_exp);
        float v_res = m1 / 2 * v_exp;
        if (v<0){v_res = v_res * -1;}
        v_res = v_res * scale;
        res[i] = v_res;
    }

}


__global__ void nvf4_quant_cuda_kernel(const float* x_ori, float* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 16 * 256;
    const float* x_offset = x_ori + offset;
    float* res_mem = res_ori + offset;

    __shared__ float x_shared[16*256];
    __shared__ float res_shared[16*256];

    for (int i=0;i<4;++i){
        if ((offset + i*block_size + thread_idx) >= n_total){
            ((float4*) x_shared)[i*block_size+thread_idx] = make_float4(0,0,0,0);
        }else{
            ((float4*) x_shared)[i*block_size+thread_idx] = ((float4*)x_offset)[i*block_size+thread_idx];
        }
    }
    __syncthreads();
    
    nvf4_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<4;++i){
        if ((offset + i*block_size + thread_idx) < n_total){
            ((float4*)res_mem)[i*block_size+thread_idx] = ((float4*)res_shared)[i*block_size+thread_idx];
        }
    }
}


__global__ void nvf4_quant_cuda_bf16_kernel(const bf16* x_ori, bf16* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * 16 * 256;
    const bf16* x_offset = x_ori + offset;
    bf16* res_mem = res_ori + offset;

    __shared__ float x_shared[16*256];
    __shared__ float res_shared[16*256];

    for (int i=0;i<4;++i){
        load_to_shared(x_offset, x_shared, offset, i*block_size+thread_idx, n_total);
    }
    __syncthreads();
    
    nvf4_quant_cuda_inner(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<4;++i){
        store_to_shared(res_mem, res_shared, offset, i*block_size+thread_idx, n_total);
    }
}


void nvf4_quant_cuda(const torch::Tensor x, torch::Tensor result){
    int threads = 256;
    int blocks = (x.numel() + 256*16 - 1) / (256*16);
    
    nvf4_quant_cuda_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        result.data_ptr<float>(),
        x.numel()
    );
}


void nvf4_quant_cuda_bf16(const torch::Tensor x, torch::Tensor result){
    int threads = 256;
    int blocks = (x.numel() + 256*16 - 1) / (256*16);
    
    nvf4_quant_cuda_bf16_kernel<<<blocks, threads>>>(
        x.data_ptr<bf16>(),
        result.data_ptr<bf16>(),
        x.numel()
    );
}
