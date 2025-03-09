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

void mm(
    cublasHandle_t handle,
    const int32_t batch_size,
    const int32_t input_size,
    const int32_t output_size,
    const float* A,
    const float* x,
    float* y)
{
    const int32_t m = output_size;
    const int32_t n = batch_size;
    const int32_t k = input_size;

    const int32_t ldA = output_size;
    const int32_t ldx = input_size;
    const int32_t ldy = output_size;

    const float alpha = 1.0F;
    const float beta = 0.0F;

    cublasSgemm_v2(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, ldA,
        x, ldx,
        &beta,
        y, ldy
    );
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
    const size_t threads = 512;
    const size_t blocks = (output_size * batch_size + threads - 1) / threads;

    mm(handle, batch_size, input_size, output_size * buckets, A, x, (float*)intmdt);
    select<<<blocks, threads>>>(
        batch_size,
        output_size * buckets,
        output_size,
        s,
        (float*)intmdt,
        y
    );
}