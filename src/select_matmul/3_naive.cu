#include "cublas_v2.h"

template<const int32_t BK>
__global__ void naive_select_matmul(
    const int32_t m,
    const int32_t n,
    const int32_t k,
    const int32_t b,
    const float* A,
    const float* X,
    const int32_t* S,
    float* Y)
{
    const int32_t row = blockIdx.x * BK + (threadIdx.x % BK);
    const int32_t col = blockIdx.y * BK + (threadIdx.x / BK);
    
    if (row < m && col < n)
    {
        const float* tA = A + m * S[col] + row;
        const float* tX = X + k * col;
        float sum = 0.0;

        for (int32_t i = 0; i < k; i++)
        {
            sum += tA[m * b * i] * tX[i];
        }

        Y[m * col + row] = sum;
    }
}

void NaiveSelectMatmul(
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
    dim3 grid((output_size + 31) / 32, (batch_size + 31) / 32);

    naive_select_matmul<32><<<grid, 1024>>>(
        output_size,
        batch_size,
        input_size,
        buckets,
        (const float*) A,
        (const float*) x,
        (const int32_t*) s,
        y
    );
}