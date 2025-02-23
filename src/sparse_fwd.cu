#include "util.cu"

__global__ void sparseKernel1(const size_t m, const size_t nnz, const float* A, const int32_t* x, float* y)
{
    const int row = threadIdx.x + blockDim.x * blockIdx.x;
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

// this kernel only works when output_size is a multiple of 4
__global__ void sparseKernel2(const size_t m, const size_t nnz, const float* A, const int32_t* x, float* y)
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

int main()
{
    const size_t inputs = 768;
    const size_t outputs = 2048;
    const size_t batch_size = 16384;
    const size_t max_active = 32;

    const size_t threadsPerBlock = 512;
    const size_t reps = 32;

    size_t maxActiveThreads;
    size_t numSMs;
    std::tie(maxActiveThreads, numSMs) = preamble();

    std::cout << "Initialising input data" << std::endl;

    std::cout << "Initialising inputs on the GPU" << std::endl;

    std::vector<float> weights = random_array<float>(inputs * outputs);

    float* weights_gpu;
    cudaMalloc((void **)&weights_gpu, sizeof(float) * inputs * outputs);
    cudaMemcpy((void *)weights_gpu, (void *)weights.data(), sizeof(float) * inputs * outputs, cudaMemcpyHostToDevice);

    std::vector<int32_t> input = random_sparse(max_active, batch_size);

    int32_t* in;
    cudaMalloc((void **)&in, sizeof(int32_t) * max_active * batch_size);
    cudaMemcpy((void *)in, (void *)input.data(), sizeof(int32_t) * max_active * batch_size, cudaMemcpyHostToDevice);

    std::cout << "Allocating output on the GPU" << std::endl;

    float* out;
    cudaMalloc((void **)&out, sizeof(float) * outputs * batch_size);

    cudaDeviceSynchronize();

    std::cout << "Running sparseKernel1" << std::endl;

    for (size_t i = 0; i < reps; i++)
    {
        const size_t numChunks = (outputs + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid(numChunks, batch_size);
        sparseKernel1<<<grid, threadsPerBlock>>>(outputs, max_active, weights_gpu, in, out);
    }

    float *expected = new float[outputs * batch_size];
    cudaMemcpy((void *)expected, (const void *)out, sizeof(float) * outputs * batch_size, cudaMemcpyDeviceToHost);
    cudaMemset(out, 0, sizeof(float) * outputs * batch_size);
    cudaDeviceSynchronize();

    std::cout << "Running sparseKernel2" << std::endl;

    for (size_t i = 0; i < reps; i++)
    {
        const size_t output_chunks = (outputs + 3) / 4;
        const size_t numChunks = (output_chunks + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid(numChunks, batch_size);
        sparseKernel2<<<grid, threadsPerBlock>>>(outputs, max_active, weights_gpu, in, out);
    }

    check_equal(outputs * batch_size, expected, out);
    cudaDeviceSynchronize();

    std::cout << "Done" << std::endl;

    cudaDeviceReset();

    delete expected;

    return 0;
}
