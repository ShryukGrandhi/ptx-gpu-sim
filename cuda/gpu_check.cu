// Runs the CUDA versions of the kernels on a real NVIDIA GPU, checks them
// against CPU references and times them with CUDA events. Useful for
// comparing the simulator's relative rankings with hardware.
//
//   nvcc -O3 -o gpu_check cuda/gpu_check.cu cuda/kernels.cu
//   ./gpu_check
//
// CI only compiles this file (GitHub-hosted runners have no GPU).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void reduce_interleaved(const float*, float*, unsigned);
extern "C" __global__ void reduce_strided_index(const float*, float*, unsigned);
extern "C" __global__ void reduce_sequential(const float*, float*, unsigned);
extern "C" __global__ void reduce_warp_shuffle(const float*, float*, unsigned);
extern "C" __global__ void transpose_naive(const float*, float*, unsigned);
extern "C" __global__ void transpose_smem(const float*, float*, unsigned);
extern "C" __global__ void transpose_smem_padded(const float*, float*, unsigned);
extern "C" __global__ void matmul_naive(const float*, const float*, float*, unsigned);
extern "C" __global__ void matmul_tiled(const float*, const float*, float*, unsigned);

#define CUDA_CHECK(x)                                                                          \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));   \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

static std::vector<float> random_floats(size_t n, unsigned seed) {
    std::vector<float> v(n);
    unsigned long long s = seed;
    for (auto& x : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = ((s >> 33) & 0xffffff) / float(0x1000000) * 2.0f - 1.0f;
    }
    return v;
}

static float* to_device(const std::vector<float>& h) {
    float* d;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

template <class Launch>
static float time_ms(Launch launch, int reps = 20) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    launch();  // warm-up
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}

static int failures = 0;
static void report(const char* name, float ms, bool ok) {
    std::printf("%-24s %9.4f ms  %s\n", name, ms, ok ? "ok" : "WRONG");
    if (!ok) ++failures;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    {  // reductions
        const unsigned n = 1 << 24, block = 256, grid = n / block;
        auto h = random_floats(n, 5);
        float* din = to_device(h);
        float* dout;
        CUDA_CHECK(cudaMalloc(&dout, grid * sizeof(float)));
        struct { const char* name; void (*k)(const float*, float*, unsigned); } ks[] = {
            {"reduce_interleaved", reduce_interleaved}, {"reduce_strided_index", reduce_strided_index},
            {"reduce_sequential", reduce_sequential}, {"reduce_warp_shuffle", reduce_warp_shuffle}};
        for (auto& k : ks) {
            float ms = time_ms([&] { k.k<<<grid, block>>>(din, dout, n); });
            std::vector<float> out(grid);
            CUDA_CHECK(cudaMemcpy(out.data(), dout, grid * sizeof(float), cudaMemcpyDeviceToHost));
            bool ok = true;
            for (unsigned b = 0; b < grid && ok; ++b) {
                double ref = 0;
                for (unsigned i = b * block; i < (b + 1) * block; ++i) ref += h[i];
                ok = std::fabs(out[b] - ref) < 1e-3;
            }
            report(k.name, ms, ok);
        }
        cudaFree(din);
        cudaFree(dout);
    }
    {  // transposes
        const unsigned n = 4096;
        auto h = random_floats((size_t)n * n, 6);
        float* din = to_device(h);
        float* dout;
        CUDA_CHECK(cudaMalloc(&dout, (size_t)n * n * sizeof(float)));
        struct { const char* name; void (*k)(const float*, float*, unsigned); } ks[] = {
            {"transpose_naive", transpose_naive}, {"transpose_smem", transpose_smem},
            {"transpose_smem_padded", transpose_smem_padded}};
        for (auto& k : ks) {
            float ms = time_ms([&] { k.k<<<dim3(n / 32, n / 32), dim3(32, 8)>>>(din, dout, n); });
            std::vector<float> out((size_t)n * n);
            CUDA_CHECK(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
            bool ok = true;
            for (unsigned y = 0; y < n && ok; y += 7)
                for (unsigned x = 0; x < n && ok; x += 5) ok = out[(size_t)x * n + y] == h[(size_t)y * n + x];
            report(k.name, ms, ok);
        }
        cudaFree(din);
        cudaFree(dout);
    }
    {  // SGEMM
        const unsigned n = 1024;
        auto ha = random_floats((size_t)n * n, 7), hb = random_floats((size_t)n * n, 8);
        float *da = to_device(ha), *db = to_device(hb), *dc;
        CUDA_CHECK(cudaMalloc(&dc, (size_t)n * n * sizeof(float)));
        struct { const char* name; void (*k)(const float*, const float*, float*, unsigned); } ks[] = {
            {"matmul_naive", matmul_naive}, {"matmul_tiled", matmul_tiled}};
        for (auto& k : ks) {
            float ms = time_ms([&] { k.k<<<dim3(n / 16, n / 16), dim3(16, 16)>>>(da, db, dc, n); }, 5);
            std::vector<float> c((size_t)n * n);
            CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
            bool ok = true;
            for (unsigned i = 0; i < n && ok; i += 97)
                for (unsigned j = 0; j < n && ok; j += 89) {
                    double ref = 0;
                    for (unsigned kk = 0; kk < n; ++kk) ref += (double)ha[(size_t)i * n + kk] * hb[(size_t)kk * n + j];
                    ok = std::fabs(c[(size_t)i * n + j] - ref) < 1e-2;
                }
            report(k.name, ms, ok);
        }
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }
    return failures ? 1 : 0;
}
