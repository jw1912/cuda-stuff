#include "cublas_v2.h"

__global__ void ptrs(const int32_t size, const int32_t stride, const int32_t* s, float* base_ptr, float* *out)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        const int32_t offset = (s == nullptr) ? tid : s[tid];
        out[tid] = base_ptr + stride * offset;
    }
}

void bmv(
    cublasHandle_t handle, 
    const int32_t batch_size,
    const int32_t input_size,
    const int32_t output_size,
    const int32_t ldA,
    float** A,
    float** x,
    float** y)
{
    const int32_t m = output_size;
    const int32_t n = input_size;

    const float alpha = 1.0F;
    const float beta = 0.0F;

    cublasSgemvBatched(
        handle,
        CUBLAS_OP_N,
        m, n,
        &alpha,
        A, ldA,
        x, 1,
        &beta,
        y, 1,
        batch_size
    );
}

void PtrsThenBMV(
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
    // controls the layout of matrix A
    // if true then result should match Matmul + Select
    // if false then results will not match
    const bool interleave = true;

    float** As = (float**) intmdt;
    float** xs = As + batch_size;
    float** ys = xs + batch_size;

    const int32_t stride = (interleave) ? output_size : output_size * buckets;
    const int32_t ldA = (interleave) ? output_size * buckets : output_size;

    const size_t threads = 512;
    const size_t blocks = (batch_size + threads - 1) / threads;

    ptrs<<<blocks, threads>>>(batch_size, stride, s, A, As);
    ptrs<<<blocks, threads>>>(batch_size, input_size, nullptr, x, xs);
    ptrs<<<blocks, threads>>>(batch_size, output_size, nullptr, y, ys);
    bmv(handle, batch_size, input_size, output_size, ldA, As, xs, ys);
}
