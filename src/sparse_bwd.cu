#include "util.cu"

__global__ void sparseKernel1(
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

__global__ void sparseKernel2(
    const int32_t m,
    const int32_t nnz,
    const float* A,
    const int32_t* B,
    float* C)
{
    const int32_t row = blockIdx.y;
    const int32_t col = threadIdx.x + blockDim.x * blockIdx.x;

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

    std::vector<float> A_cpu = random_array<float>(batch_size * outputs);

    float* A;
    cudaMalloc((void **)&A, sizeof(float) * batch_size * outputs);
    cudaMemcpy((void *)A, (void *)A_cpu.data(), sizeof(float) * batch_size * outputs, cudaMemcpyHostToDevice);

    std::vector<int32_t> B_cpu = random_sparse(max_active, batch_size);

    int32_t* B;
    cudaMalloc((void **)&B, sizeof(int32_t) * max_active * batch_size);
    cudaMemcpy((void *)B, (void *)B_cpu.data(), sizeof(int32_t) * max_active * batch_size, cudaMemcpyHostToDevice);

    std::cout << "Allocating output on the GPU" << std::endl;

    float* C;
    const size_t C_size = sizeof(float) * inputs * outputs;
    cudaMalloc((void **)&C, C_size);
    cudaMemset(C, 0, C_size);

    cudaDeviceSynchronize();

    std::cout << "Running sparseKernel1" << std::endl;

    for (size_t i = 0; i < reps; i++)
    {
        const size_t numChunks = (outputs + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid(numChunks, batch_size);
        sparseKernel1<<<grid, threadsPerBlock>>>(outputs, max_active, A, B, C);
    }

    float *expected = new float[outputs * batch_size];
    cudaMemcpy((void *)expected, (const void *)C, C_size, cudaMemcpyDeviceToHost);
    cudaMemset(C, 0, C_size);
    cudaDeviceSynchronize();

    std::cout << "Running sparseKernel2" << std::endl;

    for (size_t i = 0; i < reps; i++)
    {
        const size_t numChunks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid(numChunks, outputs);
        sparseKernel2<<<grid, threadsPerBlock>>>(outputs, max_active, A, B, C);
    }

    check_equal(inputs * outputs, expected, C);
    cudaDeviceSynchronize();

    std::cout << "Done" << std::endl;

    cudaDeviceReset();

    delete expected;

    return 0;
}
