#include "util.cu"

__host__ __device__ float relu(const float in)
{
    return max(in, 0.0F);
}

__global__ void reluKernel1(const size_t size, const float *in, float *out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        out[tid] = relu(in[tid]);
    }
}

__global__ void reluKernel2(const size_t size, const float *in, float *out)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; tid < size; tid += gridDim.x * blockDim.x)
    {
        out[tid] = relu(in[tid]);
    }
}

__global__ void reluKernel3(const size_t size, const float *in, float *out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 a = ((const float4 *)in)[tid];
        ((float4 *)out)[tid] = make_float4(relu(a.x), relu(a.y), relu(a.z), relu(a.w));
    }
    else if (4 * tid < size)
    {
        for (size_t i = 0; i < size - 4 * tid; i++)
        {
            const size_t idx = 4 * tid + i;
            out[idx] = relu(in[idx]);
        }
    }
}

int main()
{
    const size_t size = 2Ui64 << 24;

    const size_t threadsPerBlock = 512;

    size_t maxActiveThreads;
    size_t numSMs;
    std::tie(maxActiveThreads, numSMs) = preamble();

    std::cout << "Initialising input data" << std::endl;

    std::cout << "Initialising inputs on the GPU" << std::endl;

    std::vector<float> inputs = random_array(size);

    float* in;
    cudaMalloc((void **)&in, sizeof(float) * size);
    cudaMemcpy((void *)in, (void *)inputs.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

    std::cout << "Allocating output on the GPU" << std::endl;

    float* out;
    cudaMalloc((void **)&out, sizeof(float) * size);

    cudaDeviceSynchronize();

    std::cout << "Calculating expected output" << std::endl;

    for (size_t i = 0; i < size; i++)
    {
        inputs[i] = relu(inputs[i]);
    }

    std::cout << "Running reluKernel1" << std::endl;

    for (size_t i = 0; i < 10; i++)
    {
        const size_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        reluKernel1<<<blocks, threadsPerBlock>>>(size, in, out);
    }

    check_equal(size, inputs.data(), out);
    cudaMemset(out, 0, size * sizeof(float));

    std::cout << "Running reluKernel2" << std::endl;

    for (size_t i = 0; i < 10; i++)
    {
        const size_t blocks = numSMs * maxActiveThreads / threadsPerBlock;
        reluKernel2<<<blocks, threadsPerBlock>>>(size, in, out);
    }

    check_equal(size, inputs.data(), out);
    cudaMemset(out, 0, size * sizeof(float));

    std::cout << "Running reluKernel3" << std::endl;

    for (size_t i = 0; i < 10; i++)
    {
        const size_t float4_size = (size + 3) / 4;
        const size_t blocks = (float4_size + threadsPerBlock - 1) / threadsPerBlock;
        reluKernel3<<<blocks, threadsPerBlock>>>(size, in, out);
    }

    check_equal(size, inputs.data(), out);

    std::cout << "Done" << std::endl;

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
