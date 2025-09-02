#include <cuda_runtime.h>
#include <cmath>

extern "C" __global__ void cosine_scores_kernel(const float* __restrict__ X, // N x D
                                                 const float* __restrict__ q, // D (assumed normalized)
                                                 float* __restrict__ out,
                                                 int N, int D){
    int i = blockIdx.x; // row idx
    if (i >= N) return;
    float acc = 0.f;
    for (int t = threadIdx.x; t < D; t += blockDim.x) {
        acc += X[i*D + t] * q[t];
    }
    // intra-block reduction
    __shared__ float sbuf[256];
    int lane = threadIdx.x;
    sbuf[lane] = acc;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1){
        if (lane < s) sbuf[lane] += sbuf[lane + s];
        __syncthreads();
    }
    if (lane == 0) out[i] = sbuf[0];
}

extern "C" __global__ void l2_scores_kernel(const float* __restrict__ X,
                                             const float* __restrict__ q,
                                             float* __restrict__ out,
                                             int N, int D){
    int i = blockIdx.x; if (i>=N) return;
    float acc = 0.f;
    for (int t = threadIdx.x; t < D; t += blockDim.x){
        float diff = X[i*D + t] - q[t];
        acc += diff*diff;
    }
    __shared__ float sbuf[256];
    int lane = threadIdx.x;
    sbuf[lane] = acc;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1){
        if (lane < s) sbuf[lane] += sbuf[lane + s];
        __syncthreads();
    }
    if (lane == 0) out[i] = -sbuf[0]; // negative to treat as similarity
}