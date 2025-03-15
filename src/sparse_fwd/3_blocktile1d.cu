// this kernel only works when `m` is a multiple of TM
template<const int32_t TM>
__global__ void sparse_fwd_blocktiled_kernel(const int32_t m, const int32_t nnz, const float4* A, const int32_t* x, float4* y)
{
    const int32_t base_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (TM * base_row < m)
    {
        float4 val[TM] = {};
        const int32_t* tx = x + nnz * blockIdx.y;
    
        for (int32_t i = 0; i < nnz; i++) {
            const int32_t inp = tx[i];

            if (inp != -1)
            {
                for (int32_t trow = 0; trow < TM; trow++)
                {
                    const float4 a = A[inp * m + base_row + trow * (m / TM)];
                    val[trow].x += a.x;
                    val[trow].y += a.y;
                    val[trow].z += a.z;
                    val[trow].w += a.w;
                }
            }
        }
    
        for (int32_t trow = 0; trow < TM; trow++)
        {
            y[m * blockIdx.y + base_row + trow * (m / TM)] = val[trow];
        }
    }
}