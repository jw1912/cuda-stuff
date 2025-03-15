__global__ void sparse_fwd_vectorised_kernel(const int32_t m, const int32_t nnz, const float4* A, const int32_t* x, float4* y)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m)
    {
        float4 val = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        const int32_t* tx = x + nnz * blockIdx.y;
    
        for (int32_t i = 0; i < nnz; i++) {
            const int32_t inp = tx[i];
    
            if (inp != -1)
            {
                const float4 a = A[inp * m + row];
                val.x += a.x;
                val.y += a.y;
                val.z += a.z;
                val.w += a.w;
            }
        }

        y[m * blockIdx.y + row] = val;
    }
}
