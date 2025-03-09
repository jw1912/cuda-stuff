#include "1_mm_select.cu"
#include "2_ptrs_bmv.cu"
#include "../util.cu"

static constexpr size_t inputs = 1024;
static constexpr size_t outputs = 16;
static constexpr size_t buckets = 8;
static constexpr size_t batch_size = 16384;
static constexpr size_t reps = 64;

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

    cublasDestroy_v2(handle);
    cudaDeviceReset();

    delete expected;

    return 0;
}