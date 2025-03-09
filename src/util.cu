#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <tuple>
#include <vector>

#define RUN_BODY(body) \
    cudaDeviceSynchronize();\
    auto t1 = std::chrono::high_resolution_clock::now();\
    for (size_t i = 0; i < reps; i++)\
    {\
        body;\
    }\
    cudaDeviceSynchronize();\
    auto t2 = std::chrono::high_resolution_clock::now();\
    std::chrono::duration<double, std::milli>  time = t2 - t1;\
    std::cout << "Average Time: ";\
    std::cout << std::setprecision(3) << std::fixed;\
    std::cout << time.count() / (double)reps << "ms";\
    std::cout << std::endl;\

std::vector<float> random_dense(size_t size, float start = -1.0F, float end = 1.0F)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(start, end);
    std::vector<float> inputs = {};
    inputs.reserve(size);

    for (size_t i = 0; i < size; i++)
    {
        inputs.push_back(dist(gen));
    }

    return inputs;
}

std::vector<int32_t> random_sparse(size_t nnz, size_t batch_size, int32_t start = -48, int32_t end = 767)
{
    std::default_random_engine gen;
    std::uniform_int_distribution<int32_t> dist(start, end);
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
        if (abs(cpu[i] - arr[i]) > 0.01)
        {
            std::cout << "Arrays don't match at index " << i << ": " << cpu[i] << " != " << arr[i] << std::endl;
            return;
        }
    }

    delete arr;
}

template<typename T>
T* init(const std::vector<T>& values)
{
    T* x;
    const size_t size = values.size();
    cudaMalloc((void **)&x, sizeof(T) * size);
    cudaMemcpy((void *)x, (void *)values.data(), sizeof(T) * size, cudaMemcpyHostToDevice);
    return x;
}
