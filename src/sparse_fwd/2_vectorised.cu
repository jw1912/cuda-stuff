// this kernel only works when `m` is a multiple of 4
__global__ void sparse_fwd_vectorised_kernel(const size_t m, const size_t nnz, const float* A, const int32_t* x, float* y)
{
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * row < m)
    {
    
        float val[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        const int32_t* tx = x + nnz * blockIdx.y;
    
        for (size_t i = 0; i < nnz; i++) {
            const int32_t inp = tx[i];
    
            if (inp != -1)
            {
                const size_t our_idx = static_cast<size_t>(inp) * m / 4;
                const float4 a = ((const float4 *)A)[our_idx + row];
                val[0] += a.x;
                val[1] += a.y;
                val[2] += a.z;
                val[3] += a.w;
            }
        }
    
        ((float4 *)y)[m * blockIdx.y / 4 + row] = make_float4(val[0], val[1], val[2], val[3]);
    }
}
