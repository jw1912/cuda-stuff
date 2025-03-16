#include "1_naive.cu"
#include "2_vectorised.cu"
#include "3_blocktile1d.cu"
#include "../util.cu"

constexpr size_t inputs = 768;
constexpr size_t outputs = 2048;
constexpr size_t batch_size = 16384;
constexpr size_t max_active = 32;
constexpr size_t threads = 512;
constexpr size_t reps = 32;

typedef void(*OpType)(const float*, const int32_t*, float*);

template<OpType op>
void run(const float* A, const int32_t* x, float* y)
{
    RUN_BODY(op(A, x, y))
}

void sparse_fwd_naive(const float* A, const int32_t* x, float* y)
{
    const size_t numChunks = (outputs + threads - 1) / threads;
    dim3 grid(numChunks, batch_size);
    sparse_fwd_naive_kernel<<<grid, threads>>>(outputs, max_active, A, x, y);
}

void sparse_fwd_vectorised(const float* A, const int32_t* x, float* y)
{
    const size_t output_chunks = (outputs + 3) / 4;
    const size_t chunks = (output_chunks + threads - 1) / threads;
    dim3 grid(chunks, batch_size);
    sparse_fwd_vectorised_kernel<<<grid, threads>>>(
        outputs / 4, 
        max_active, 
        reinterpret_cast<const float4 *>(A), 
        x, 
        reinterpret_cast<float4 *>(y)
    );
}

void sparse_fwd_blocktiled(const float* A, const int32_t* x, float* y)
{
    constexpr int32_t tm = 8;
    const size_t output_chunks = (outputs + (4 * tm - 1)) / (4 * tm);
    const size_t req_threads = min(threads, output_chunks);
    const size_t chunks = (output_chunks + req_threads - 1) / req_threads;
    dim3 grid(chunks, batch_size);
    sparse_fwd_blocktiled_kernel<tm><<<grid, req_threads>>>(
        outputs / 4,
        max_active,
        reinterpret_cast<const float4 *>(A),
        x,
        reinterpret_cast<float4 *>(y)
    );
}

int main()
{
    float* A = init(random_dense(inputs * outputs));
    int32_t* x = init(random_sparse(max_active, batch_size));
    float* y;
    float *expected = new float[outputs * batch_size];

    cudaMalloc((void **)&y, sizeof(float) * outputs * batch_size);
    cudaDeviceSynchronize();

    std::cout << "Running naive" << std::endl;
    run<sparse_fwd_naive>(A, x, y);
    check_error();
    cudaMemcpy((void *)expected, (const void *)y, sizeof(float) * outputs * batch_size, cudaMemcpyDeviceToHost);
    cudaMemset((void*) y, 0, sizeof(float) * outputs * batch_size);
    check_error();
    std::cout << "Running vectorised" << std::endl;
    run<sparse_fwd_vectorised>(A, x, y);
    check_error();
    check_equal(outputs * batch_size, expected, y);
    cudaMemset((void*) y, 0, sizeof(float) * outputs * batch_size);
    check_error();
    std::cout << "Running blocktiled" << std::endl;
    run<sparse_fwd_blocktiled>(A, x, y);
    check_error();
    check_equal(outputs * batch_size, expected, y);
    cudaMemset((void*) y, 0, sizeof(float) * outputs * batch_size);
    check_error();
    cudaDeviceReset();

    delete expected;

    return 0;
}
