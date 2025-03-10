#include "1_mm_select.cu"
#include "2_ptrs_bmv.cu"
#include "3_naive.cu"
#include "4_cache_smem.cu"
#include "../util.cu"

// must be a multiple of 32
constexpr size_t inputs = 1024;

// arbitrary nonzero
constexpr size_t outputs = 16;
constexpr size_t buckets = 8;
constexpr size_t batch_size = 16384;
constexpr size_t reps = 64;

typedef void(*OpType)(cublasHandle_t, int32_t, int32_t, int32_t, int32_t, float*, float*, int32_t*, float*, void*);

template<OpType op>
void run(cublasHandle_t handle, float* A, float* x, int32_t* s, float* y, void* intmdt)
{
    RUN_BODY(op(handle, batch_size, inputs, outputs, buckets, A, x, s, y, intmdt))
}

int main()
{
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    float* A = init(random_dense(inputs * outputs * buckets));
    float* x = init(random_dense(inputs * batch_size));
    int32_t* s = init(random_sparse(1, batch_size, 0, buckets - 1));
    float* y;
    void* intmdt;
    float *expected = new float[outputs * batch_size];

    cudaMalloc((void **)&y, sizeof(float) * outputs * batch_size);
    cudaMalloc(&intmdt, sizeof(float) * outputs * buckets * batch_size);

    std::cout << "Running Matmul + Select" << std::endl;
    run<MatmulThenSelect>(handle, A, x, s, y, intmdt);
    cudaMemcpy((void *)expected, (const void *)y, sizeof(float) * outputs * batch_size, cudaMemcpyDeviceToHost);

    std::cout << "Running Broadcast Ptrs + Batched Matmul" << std::endl;
    run<PtrsThenBMV>(handle, A, x, s, y, intmdt);
    check_equal(outputs * batch_size, expected, y);
    cudaMemset((void*) y, 0, sizeof(float) * outputs * batch_size);

    std::cout << "Running Naive Select Matmul" << std::endl;
    run<NaiveSelectMatmul>(handle, A, x, s, y, intmdt);
    check_equal(outputs * batch_size, expected, y);
    cudaMemset((void*) y, 0, sizeof(float) * outputs * batch_size);

    std::cout << "Running Cached SMEM Select Matmul" << std::endl;
    run<CacheSMEMSelectMatmul>(handle, A, x, s, y, intmdt);
    check_equal(outputs * batch_size, expected, y);

    cublasDestroy_v2(handle);
    cudaDeviceReset();

    delete expected;

    return 0;
}