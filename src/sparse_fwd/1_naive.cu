__global__ void sparse_fwd_naive_kernel(const int32_t m, const int32_t nnz, const float* A, const int32_t* x, float* y)
{
    const int32_t row = threadIdx.x + blockDim.x * blockIdx.x;
    const int32_t* tx = x + nnz * blockIdx.y;
    float* ty = y + m * blockIdx.y; 

    if (row < m)
    {
        float sum = 0;
        for (int i = 0; i < nnz; i += 1)
        {
            const int j = tx[i];
            if (j != -1)
            {
                sum += A[m * j + row];
            }
        }

        ty[row] = sum;
    }
}
