#include <cmath>
#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/Atomic.cuh>


using bf16 = at::BFloat16;
using athalf = at::Half;


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



__device__ __forceinline__ float extract_exp(float x){
    int a = *(int*)&x;
    a = a & 0x7F800000;
    float res = *(float*)&a;
    return res;
}


template <int N>
constexpr int get_mask(){
    if constexpr (N==4){
        return 0x7FF80000;
    }
    if constexpr (N==3){
        return 0x7FF00000;
    }
    if constexpr (N==2){
        return 0x7FE00000;
    }
    if constexpr (N==1){
        return 0x7FC00000;
    }
    if constexpr (N==0){
        return 0x7F800000;
    }
}


template <int N>
__device__ __forceinline__ float round_mant(float x, float e){
    // There is another 
    // float add is a good approx, but sometimes will result in wrong results. E.g. 0xBC7FFFE2 + 0.5f
    int x_int = *(int*) &x;
    int x_int_sign = (x_int & 0x80000000) | 0x3F800000;
    int x_int_unsigned = x_int & 0x7FFFFFFF;

    float x_unsigned = *(float*) &x_int_unsigned;
    float x_sign = *(float*) &x_int_sign;

    float mant = x_unsigned / e;
    float mant2 = mant * (float)(1<<N);
    float mant3 = std::floor(mant2 + 0.5);
    // printf("x:%f  m: %f  m2: %f  m3: %f\n", x, mant, mant2, mant3);
    if (mant3 == (float)(1<<(N+1))){
        mant3 = mant3 - 1;
    }
    mant3 = mant3 / (float)(1<<N);
    float out = x_sign * e * mant3;
    return out;
}


template <typename T>
__global__ void hif8_quant_cuda_kernel(const T* x_ori, T* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * block_size;
    const T* x_offset = x_ori + offset;
    T* res_mem = res_ori + offset;


    // memory read 
    if (offset + thread_idx < n_total){
        float x = (float) x_offset[thread_idx];
        float absx = abs(x);
        if (absx <= (1.0f/(float)(1<<23))){
            res_mem[thread_idx] = 0;
        }else if(x<=-40960.0f){
            uint neginf_int = 0xFF800000;
            float neginf = *(float*) &neginf_int;
            res_mem[thread_idx] = neginf;
        }else if(x>=40960.0f){
            uint posinf_int = 0x7F800000;
            float posinf = *(float*) &posinf_int;
            res_mem[thread_idx] = posinf;
        }
        else{
            uint x_int = *(uint*)&x;
            uint exp_x = (x_int & 0x7F800000);
            if (exp_x==0x34000000){
                exp_x = 0x34800000;
            }
            
            float exp_bias = 1.0f;  // 2**mant_bits
            if ((exp_x<=0x47000000) && (exp_x>=0x38000000)){
                exp_bias = 2.0f;
            }
            if ((exp_x<=0x43000000) && (exp_x>=0x3C000000)){
                exp_bias = 4.0f;
            }
            if ((exp_x<=0x41000000) && (exp_x>=0x3E000000)){
                exp_bias = 8.0f;
            }

            // round mantissa 
            float exp_f = *(float*) &exp_x;
            float mant = absx / exp_f * exp_bias;
            mant = floor(mant + 0.5);
            mant = mant * exp_f / exp_bias;
            float signx = (x>=0) ? (1.0f) : (-1.0f);
            float res = signx * mant;
            res_mem[thread_idx] = (T) res;
        }
    }
}


template <int N>
__device__ __forceinline__ float round_mant_without_sat_to_even(float x, float e){
    // There is another 
    // float add is a good approx, but sometimes will result in wrong results. E.g. 0xBC7FFFE2 + 0.5f
    int x_int = *(int*) &x;
    int x_int_sign = (x_int & 0x80000000) | 0x3F800000;
    int x_int_unsigned = x_int & 0x7FFFFFFF;

    float x_unsigned = *(float*) &x_int_unsigned;
    float x_sign = *(float*) &x_int_sign;

    float mant = x_unsigned / e;
    float mant2 = mant * (float)(1<<N);
    float mant3 = std::round(mant2);
    mant3 = mant3 / (float)(1<<N);
    float out = x_sign * e * mant3;
    return out;
}

template <int N>
__device__ __forceinline__ float round_mant(float x, float e, float lv0_rec, float lv0){
    // There is another 
    // float add is a good approx, but sometimes will result in wrong results. E.g. 0xBC7FFFE2 + 0.5f
    int x_int = *(int*) &x;
    int x_int_sign = (x_int & 0x80000000) | 0x3F800000;
    int x_int_unsigned = x_int & 0x7FFFFFFF;

    float x_unsigned = *(float*) &x_int_unsigned;
    float x_sign = *(float*) &x_int_sign;

    float mant = x_unsigned / e * lv0_rec;
    float mant2 = mant * (float)(1<<N);
    float mant3 = std::floor(mant2 + 0.5);
    
    if (mant3 >= (float)(1<<(N+1))){
        mant3 = (float)(1<<(N+1)) - 1.0f;
    }

    mant3 = mant3 / (float)(1<<N);
    float out = x_sign * e * mant3 * lv0;
    return out;
}

__device__ __forceinline__ float f32_to_bf16(float x){
    if (x==0){ return 0; }
    float e = extract_exp(x);
    float mant = x / e * 128.0f; 
    mant = nearbyintf(mant);
    mant = mant / 128.0f * e;
    return mant;
}

__device__ __forceinline__ float f32_to_e6m2(float x){
    if (x==0){ return 0; }
    float e = extract_exp(x);
    float mant = x / e * 4.0f; 
    mant = nearbyintf(mant);
    mant = mant / 4.0f * e;
    return mant;
}


template <int N>
__device__ void hifx_quant_cuda_inner(float* x_shared, float* res_shared, const int& thread_idx){
    float lv3[64], lv2[16], lv1[8], lv0;
    for (int i=0; i<64; ++i){
        lv3[i] = x_shared[thread_idx * 64 + i];
    }
    // get level 2
    for (int i=0; i<16; ++i){
        lv2[i] = abs(lv3[i*4]);
        for (int j=1; j<4; ++j){
            float e = abs(lv3[i*4+j]);
            if (e>lv2[i]){ lv2[i] = e; }
        }
    }
    // get level 1
    for (int i=0; i<8; ++i){
        lv1[i] = (lv2[i*2]>lv2[i*2+1]) ? lv2[i*2] : lv2[i*2+1];
    }
    // get level 0
    lv0 = lv1[0];
    for (int i=1; i<8; ++i){
        lv0 = (lv0 < lv1[i]) ? lv1[i] : lv0;
    }

    float inv7 = f32_to_bf16(1.0f / 7.0f);
    lv0 = lv0 * inv7;
    lv0 = (lv0 > 49152.0f) ? 49152.0f : lv0;
    float two_pow_neg48 = std::pow(2.0f, -48.0f);
    lv0 = (lv0 < two_pow_neg48) ? two_pow_neg48 : lv0; 
    lv0 = f32_to_bf16(lv0);
    
    // float lv0_exp = extract_exp(lv0);
    // lv0 = round_mant_without_sat_to_even<2>(lv0, lv0_exp);
    lv0 = f32_to_e6m2(lv0);
    
    float rec_lv0 = f32_to_bf16(1.0f / lv0);
    
    // get shared exp
    for (int i=0; i<8; ++i){
        lv1[i] = lv1[i] * rec_lv0;
        lv1[i] = (lv1[i] >= 4.0f) ? 2.0f : 1.0f;
    }

    for (int i=0; i<16; ++i){
        lv2[i] = lv2[i] / lv1[i/2] * rec_lv0 ;
        lv2[i] = (lv2[i] >= 2.0f) ? 2.0f : 1.0f;
        lv2[i] = lv1[i/2] * lv2[i];
    }
        
    // find final number 
    for (int i=0; i<16; ++i){
        lv3[i*4] = round_mant<N>(lv3[i*4], lv2[i], rec_lv0, lv0);
        lv3[i*4+1] = round_mant<N>(lv3[i*4+1], lv2[i], rec_lv0, lv0);
        lv3[i*4+2] = round_mant<N>(lv3[i*4+2], lv2[i], rec_lv0, lv0);
        lv3[i*4+3] = round_mant<N>(lv3[i*4+3], lv2[i], rec_lv0, lv0);
    }

    for (int i=0; i<64; ++i){
        res_shared[thread_idx * 64 + i] = lv3[i];
    }

    
}


template <int N>
__global__ void hifx_quant_cuda_kernel(const float* x_ori, float* res_ori, const int n_total){
    constexpr int G=32;
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx *  2 * G * block_size;
    const float* x_offset = x_ori + offset;
    float* res_mem = res_ori + offset;

    __shared__ float x_shared[4096];  // max shared_size 
    __shared__ float res_shared[4096];

    // memory read 
    for (int i=0;i<2*G;++i){
        if (offset + block_size*i + thread_idx >= n_total){
            x_shared[block_size*i + thread_idx] = 0;
        }else{
            x_shared[block_size*i + thread_idx] = x_offset[block_size*i + thread_idx];
        }
    }
    __syncthreads();
    
    hifx_quant_cuda_inner<N>(x_shared, res_shared, thread_idx);

    // write to memory (this will result in continuous write to memory, thus faster)
    __syncthreads();
    for (int i=0;i<2*G;++i){
        if (offset + block_size*i + thread_idx >= n_total){
            // res_mem[block_size*i + thread_idx] = 0;
        }else{
            res_mem[block_size*i + thread_idx] = res_shared[block_size*i + thread_idx];
        }
    }
}



void hifx_quant_cuda(torch::Tensor x, torch::Tensor result, int N){
    int threads = 4096 / 2 / 32;
    int blocks = (x.numel() + 4095) / 4096;
    
    if ((N==2)){
        hifx_quant_cuda_kernel<2><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }
    if ((N==3)){
        hifx_quant_cuda_kernel<3><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }
    if ((N==1)){
        hifx_quant_cuda_kernel<1><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }
    if ((N==0)){
        hifx_quant_cuda_kernel<0><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }

}

template <int N>
__global__ void hifx_quant_cuda_bf16_kernel(const bf16* x_ori, bf16* res_ori, const int n_total){
    constexpr int G=32;
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx *  2 * G * block_size;
    const bf16* x_offset = x_ori + offset;
    bf16* res_mem = res_ori + offset;

    __shared__ float x_shared[4096];  // max shared_size 
    __shared__ float res_shared[4096];

    for (int i=0;i<8;++i){
        load_to_shared(x_offset, x_shared, offset, i*block_size+thread_idx, n_total);
    }
    __syncthreads();
    
    hifx_quant_cuda_inner<N>(x_shared, res_shared, thread_idx);
    __syncthreads();

    // write to memory 
    for (int i=0;i<8;++i){
        store_to_shared(res_mem, res_shared, offset, i*block_size+thread_idx, n_total);
    }
}

void hifx_quant_cuda_bf16(torch::Tensor x, torch::Tensor result, int N){
    int threads = 4096 / 2 / 32;
    int blocks = (x.numel() + 4095) / 4096;
    
    if ((N==2)){
        hifx_quant_cuda_bf16_kernel<2><<<blocks, threads>>>(x.data_ptr<bf16>(), result.data_ptr<bf16>(), x.numel());
    }
    if ((N==3)){
        hifx_quant_cuda_bf16_kernel<3><<<blocks, threads>>>(x.data_ptr<bf16>(), result.data_ptr<bf16>(), x.numel());
    }
    if ((N==1)){
        hifx_quant_cuda_bf16_kernel<1><<<blocks, threads>>>(x.data_ptr<bf16>(), result.data_ptr<bf16>(), x.numel());
    }
    if ((N==0)){
        hifx_quant_cuda_bf16_kernel<0><<<blocks, threads>>>(x.data_ptr<bf16>(), result.data_ptr<bf16>(), x.numel());
    }

}