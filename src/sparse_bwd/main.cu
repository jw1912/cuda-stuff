#include "1_naive.cu"
#include "../util.cu"

constexpr size_t inputs = 768;
constexpr size_t outputs = 2048;
constexpr size_t batch_size = 16384;
constexpr size_t max_active = 32;
constexpr size_t threads = 512;
constexpr size_t reps = 32;

typedef void(*OpType)(const float*, const int32_t*, float*);

template<OpType op>
void run(const float* A, const int32_t* B, float* C)
{
    RUN_BODY(op(A, B, C))
}

void sparse_bwd_naive(const float* A, const int32_t* B, float* C)
{
    const size_t chunks = (outputs + threads - 1) / threads;
    dim3 grid(chunks, batch_size);
    sparse_bwd_naive_kernel<<<grid, threads>>>(outputs, max_active, A, B, C);
}

int main()
{
    float* A = init(random_dense(batch_size * outputs));
    int32_t* B = init(random_sparse(max_active, batch_size));
    float* C;
    float *expected = new float[outputs * batch_size];

    const size_t C_size = sizeof(float) * inputs * outputs;
    cudaMalloc((void **)&C, C_size);
    cudaMemset(C, 0, C_size);
    cudaDeviceSynchronize();

    std::cout << "Running naive" << std::endl;
    run<sparse_bwd_naive>(A, B, C);
    cudaMemcpy((void *)expected, (const void *)C, C_size, cudaMemcpyDeviceToHost);

    std::cout << "Running naive" << std::endl;
    run<sparse_bwd_naive>(A, B, C);
    check_equal(inputs * outputs, expected, C);

    cudaDeviceReset();

    delete expected;

    return 0;
}
