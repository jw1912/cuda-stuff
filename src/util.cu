#include <iostream>
#include <random>
#include <tuple>
#include <vector>

std::tuple<size_t, size_t> preamble()
{
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties_v2(&props, device);

    const size_t maxActiveThreads = props.maxThreadsDim[0];
    const size_t numSMs = props.multiProcessorCount;

    std::cout << "Max active threads per SM: " << maxActiveThreads << std::endl;
    std::cout << "Number of SMs: " << numSMs << std::endl;

    return {maxActiveThreads, numSMs};
}

template <typename T>
std::vector<T> random_array(size_t size);

template <>
std::vector<float> random_array<float>(size_t size)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> inputs = {};
    inputs.reserve(size);

    for (size_t i = 0; i < size; i++)
    {
        inputs.push_back(dist(gen));
    }

    return inputs;
}

template <>
std::vector<int32_t> random_array<int32_t>(size_t size)
{
    std::default_random_engine gen;
    std::uniform_int_distribution<int32_t> dist(-1, 767);
    std::vector<int32_t> inputs = {};
    inputs.reserve(size);

    for (size_t i = 0; i < size; i++)
    {
        inputs.push_back(dist(gen));
    }

    return inputs;
}

std::vector<int32_t> random_sparse(size_t nnz, size_t batch_size)
{
    std::default_random_engine gen;
    std::uniform_int_distribution<int32_t> dist(-48, 767);
    std::vector<int32_t> inputs = {};
    inputs.reserve(nnz * batch_size);

    for (size_t i = 0; i < batch_size; i++)
    {
        bool ended = false;
        for (size_t j = 0; j < nnz; j++)
        {
            int32_t val = ended ? -1 : dist(gen);
            val = val < 0 ? -1 : val;
            ended = val < 0;
            inputs.push_back(val);
        }
    }

    return inputs;
}

void check_error()
{
    cudaDeviceSynchronize();
    const cudaError_t err = cudaGetLastError();
    std::cout << "Error Status: " << cudaGetErrorString(err) << std::endl;
}

void check_equal(const size_t size, const float *cpu, const float *gpu)
{
    float* arr = new float[size];

    check_error();

    cudaMemcpy((void *)arr, (const void*)gpu, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < size; i++)
    {
        if (abs(cpu[i] - arr[i]) > 0.00001)
        {
            std::cout << "Arrays don't match at index " << i << ": " << cpu[i] << " != " << arr[i] << std::endl;
            return;
        }
    }

    delete arr;
}
