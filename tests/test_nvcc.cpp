// Conformance: run the PTX that `nvcc -ptx` generates from cuda/kernels.cu
// through the simulator and check results with the same drivers used for the
// hand-written kernels. Enabled with --nvcc-ptx <file>; skipped otherwise.
#include "kernel_harness.h"
#include "test.h"

using namespace harness;

static const Module& nvcc_kernels() {
    static Module m = load_kernels(testing::nvcc_ptx());
    return m;
}

#define REQUIRE_NVCC_PTX() \
    if (testing::nvcc_ptx().empty()) SKIP("no --nvcc-ptx file given")

TEST(nvcc_ptx_memory_kernels) {
    REQUIRE_NVCC_PTX();
    Gpu gpu;
    CHECK(run_vecadd(gpu, nvcc_kernels(), 5000).correct);
    CHECK(run_strided_copy(gpu, nvcc_kernels(), 4096, 3).correct);
    CHECK(run_copy_vec4(gpu, nvcc_kernels(), 8192).correct);
}

TEST(nvcc_ptx_reductions) {
    REQUIRE_NVCC_PTX();
    Gpu gpu;
    for (const char* k : {"reduce_interleaved", "reduce_strided_index", "reduce_sequential", "reduce_warp_shuffle"})
        CHECK(run_reduce(gpu, nvcc_kernels(), k, 5000).correct);
}

TEST(nvcc_ptx_transposes) {
    REQUIRE_NVCC_PTX();
    Gpu gpu;
    for (const char* k : {"transpose_naive", "transpose_smem", "transpose_smem_padded"})
        CHECK(run_transpose(gpu, nvcc_kernels(), k, 96).correct);
    RunResult smem = run_transpose(gpu, nvcc_kernels(), "transpose_smem", 64);
    RunResult pad = run_transpose(gpu, nvcc_kernels(), "transpose_smem_padded", 64);
    CHECK(smem.stats.bank_conflicts() > 0);
    CHECK_EQ(pad.stats.bank_conflicts(), (uint64_t)0);
}

TEST(nvcc_ptx_matmul_and_histogram) {
    REQUIRE_NVCC_PTX();
    Gpu gpu;
    CHECK(run_matmul(gpu, nvcc_kernels(), "matmul_naive", 48).correct);
    CHECK(run_matmul(gpu, nvcc_kernels(), "matmul_tiled", 48).correct);
    CHECK(run_histogram(gpu, nvcc_kernels(), "histogram_global", 10000, 3, 8).correct);
    CHECK(run_histogram(gpu, nvcc_kernels(), "histogram_shared", 10000, 3, 8).correct);
}
