#include "util.cu"

__global__ void sparseKernel1(const size_t max_active, const size_t output_size, const float* weights, const int32_t* inputs, float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= output_size)
        return;

    const size_t inputIdx = max_active * blockIdx.y;
    const int32_t* thisInput = inputs + inputIdx;
    float* thisOutput = outputs + output_size * blockIdx.y + elem;

    float ourElementVal = 0.0F;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * output_size + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
}

__global__ void sparseKernel2(const size_t max_active, const size_t output_size, const float* weights, const int32_t* inputs, float* outputs)
{
    extern __shared__ int32_t shared_input_indices[];
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= output_size)
        return;

    if (threadIdx.x < max_active)
    {
        const size_t input_idx = max_active * blockIdx.y;
        const int32_t* this_input = inputs + input_idx;

        for (size_t i = threadIdx.x; i < max_active; i += blockDim.x)
        {
            shared_input_indices[i] = this_input[i];
        }
    }

    __syncthreads();

    float val = 0.0F;

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = shared_input_indices[i];

        if (inp == -1)
            break;

        const size_t our_idx = static_cast<size_t>(inp) * output_size + elem;
        val += weights[our_idx];
    }

    outputs[output_size * blockIdx.y + elem] = val;
}

// this kernel only works when output_size is a multiple of 4
__global__ void sparseKernel3(const size_t max_active, const size_t output_size, const float* weights, const int32_t* inputs, float* outputs)
{
    extern __shared__ int32_t shared_input_indices[];
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * elem >= output_size)
        return;

    if (threadIdx.x < max_active)
    {
        const size_t input_idx = max_active * blockIdx.y;
        const int32_t* this_input = inputs + input_idx;

        for (size_t i = threadIdx.x; i < max_active; i += blockDim.x)
        {
            shared_input_indices[i] = this_input[i];
        }
    }

    __syncthreads();

    float val[4] = {0.0F, 0.0F, 0.0F, 0.0F};

    for (size_t i = 0; i < max_active; i++) {
        const int32_t inp = shared_input_indices[i];

        if (inp == -1)
            break;

        const size_t our_idx = static_cast<size_t>(inp) * output_size / 4;
        const float4 a = ((const float4 *)weights)[our_idx + elem];
        val[0] += a.x;
        val[1] += a.y;
        val[2] += a.z;
        val[3] += a.w;
    }

    ((float4 *)outputs)[output_size * blockIdx.y / 4 + elem] = make_float4(val[0], val[1], val[2], val[3]);
}

int main()
{
    const size_t inputs = 768;
    const size_t outputs = 2048;
    const size_t batch_size = 16384;
    const size_t max_active = 64;

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

    std::vector<int32_t> input = random_array<int32_t>(max_active * batch_size);

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
        sparseKernel1<<<grid, threadsPerBlock>>>(max_active, outputs, weights_gpu, in, out);
    }

    float *expected = new float[outputs * batch_size];
    cudaMemcpy((void *)expected, (const void *)out, sizeof(float) * outputs * batch_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    check_equal(outputs * batch_size, expected, out);
    cudaMemset(out, 0, sizeof(float) * outputs * batch_size);
    cudaDeviceSynchronize();

    std::cout << "Running sparseKernel2" << std::endl;

    for (size_t i = 0; i < reps; i++)
    {
        const size_t numChunks = (outputs + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid(numChunks, batch_size);
        sparseKernel2<<<grid, threadsPerBlock, max_active * sizeof(int32_t)>>>(max_active, outputs, weights_gpu, in, out);
    }

    check_equal(outputs * batch_size, expected, out);
    cudaMemset(out, 0, sizeof(float) * outputs * batch_size);
    cudaDeviceSynchronize();

    std::cout << "Running sparseKernel3" << std::endl;

    for (size_t i = 0; i < reps; i++)
    {
        const size_t output_chunks = (outputs + 3) / 4;
        const size_t numChunks = (output_chunks + threadsPerBlock - 1) / threadsPerBlock;
        dim3 grid(numChunks, batch_size);
        sparseKernel3<<<grid, threadsPerBlock, max_active * sizeof(int32_t) + sizeof(float) * threadsPerBlock>>>(max_active, outputs, weights_gpu, in, out);
    }

    check_equal(outputs * batch_size, expected, out);

    std::cout << "Done" << std::endl;

    cudaDeviceSynchronize();
    cudaDeviceReset();

    delete expected;

    return 0;
}
