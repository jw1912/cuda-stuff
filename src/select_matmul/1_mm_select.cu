#include "cublas_v2.h"

__global__ void select(
    const int32_t b,
    const int32_t m,
    const int32_t n,
    const int32_t* s,
    const float* x,
    float* y)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= b * n)
        return;

    const int32_t row = tid % n;
    const int32_t col = tid / n;
    
    y[n * col + row] = x[m * col + n * s[col] + row];
}

void MatmulThenSelect(
    cublasHandle_t handle,
    const int32_t batch_size,
    const int32_t input_size,
    const int32_t output_size,
    const int32_t buckets,
    float* A,
    float* x,
    int32_t* s,
    float* y,
    void* intmdt)
{
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const int32_t m = output_size * buckets;

    cublasSgemm_v2(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, batch_size, input_size,
        &alpha,
        A, m,
        x, input_size,
        &beta,
        (float*) intmdt, m
    );

    const size_t threads = 512;
    const size_t blocks = (output_size * batch_size + threads - 1) / threads;

    select<<<blocks, threads>>>(
        batch_size,
        output_size * buckets,
        output_size,
        s,
        (float*)intmdt,
        y
    );
}