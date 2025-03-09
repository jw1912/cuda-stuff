__global__ void sparse_bwd_naive_kernel(
    const int32_t m,
    const int32_t nnz,
    const float* A,
    const int32_t* B,
    float* C)
{
    const int32_t row = threadIdx.x + blockDim.x * blockIdx.x;
    const int32_t col = blockIdx.y;

    if (row < m)
    {
        const float ta = A[m * col + row];
        const int32_t* tb = B + nnz * col;
        
        for (int32_t i = 0; i < nnz; i++)
        {
            const int32_t j = tb[i];
            if (j != -1)
            {
                atomicAdd(&C[m * j + row], ta);
            }
        }
    }
}