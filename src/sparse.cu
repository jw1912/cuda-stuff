#include "util.cu"

__global__ void sparseKernel1(const size_t input_size, const size_t output_size, const float* weights, const int32_t* inputs, float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= output_size)
        return;

    const size_t inputIdx = input_size * blockIdx.y;
    const int32_t* thisInput = inputs + inputIdx;
    float* thisOutput = outputs + output_size * blockIdx.y + elem;

    float ourElementVal = 0.0F;

    for (size_t i = 0; i < input_size; i++) {
        const int32_t inp = thisInput[i];

        if (inp == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp) * output_size + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
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

    std::cout << "Running sparseKernel1" << std::endl;

    for (size_t i = 0; i < 10; i++)
    {
        const size_t blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        //sparseKernel1<<<blocks, threadsPerBlock>>>(size, in, out);
    }

    std::cout << "Done" << std::endl;

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
