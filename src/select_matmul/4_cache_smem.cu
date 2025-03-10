#include "cublas_v2.h"

template<const int32_t BK>
__global__ void cache_smem_select_matmul(
    const int32_t m,
    const int32_t n,
    const int32_t k,
    const int32_t b,
    const float* A,
    const float* X,
    const int32_t* S,
    float* Y)
{
    __shared__ float Xs[BK * BK];

    const int32_t tx = threadIdx.x % BK;
    const int32_t ty = threadIdx.x / BK;
    const int32_t row = blockIdx.x * BK + tx;
    const int32_t col = blockIdx.y * BK + ty;

    if (col < n)
    {
        A += m * S[col] + row;
        X += k * col;

        float sum = 0.0;

        for (int32_t bk = 0; bk < k; bk += BK)
        {
            Xs[ty * BK + tx] = X[bk + tx];

            __syncthreads();

            if (row < m)
            {
                for (int32_t tk = 0; tk < BK; tk++)
                {
                    const int32_t i = bk + tk;
                    sum += A[m * b * i] * Xs[ty * BK + tk];
                }
            }

            __syncthreads();
        }

        if (row < m)
        {
            Y[m * col + row] = sum;
        }
    }
}

void CacheSMEMSelectMatmul(
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

    cache_smem_select_matmul<32><<<grid, 1024>>>(
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
