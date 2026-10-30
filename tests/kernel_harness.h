// Host-side drivers for the kernels in kernels/*.ptx (and the matching CUDA
// kernels compiled by nvcc). Each driver allocates inputs, launches the kernel
// on the simulated GPU, checks the output against a CPU reference and returns
// the launch statistics. Shared by the tests and the benchmark.
#pragma once

#include "gpu.h"
#include "ptx.h"

#include <cmath>
#include <dirent.h>
#include <string>
#include <vector>

namespace harness {

using namespace ptxsim;

struct RunResult {
    KernelStats stats;
    bool correct = false;
    double max_error = 0.0;
};

// Loads and merges every .ptx file in a directory, or a single file.
inline Module load_kernels(const std::string& path) {
    Module all;
    std::vector<std::string> files;
    if (path.size() > 4 && path.substr(path.size() - 4) == ".ptx") {
        files.push_back(path);
    } else {
        DIR* d = opendir(path.c_str());
        if (!d) throw std::runtime_error("cannot open kernel directory " + path);
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".ptx") files.push_back(path + "/" + n);
        }
        closedir(d);
    }
    for (const auto& f : files) {
        Module m = load_ptx_file(f);
        for (auto& k : m.kernels) all.kernels.push_back(std::move(k));
    }
    return all;
}

// Deterministic pseudo-random inputs (no dependence on the C library's rand()).
class Lcg {
public:
    explicit Lcg(uint64_t seed) : s_(seed) {}
    uint32_t next() {
        s_ = s_ * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t)(s_ >> 33);
    }
    float uniform() { return (next() & 0xffffff) / float(0x1000000); }  // [0, 1)

private:
    uint64_t s_;
};

inline std::vector<float> random_floats(size_t n, uint64_t seed) {
    Lcg r(seed);
    std::vector<float> v(n);
    for (auto& x : v) x = r.uniform() * 2.0f - 1.0f;
    return v;
}

inline double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

inline RunResult run_vecadd(Gpu& gpu, const Module& m, uint32_t n, uint32_t block = 256,
                            const LaunchOptions& opt = LaunchOptions()) {
    gpu.memory().reset();
    auto a = random_floats(n, 1), b = random_floats(n, 2);
    uint64_t da = gpu.upload(a), db = gpu.upload(b), dc = gpu.alloc_zeroed<float>(n);
    RunResult r;
    r.stats = gpu.launch(m.kernel("vecadd"), {ceil_div(n, block)}, {block}, KernelArgs().ptr(da).ptr(db).ptr(dc).u32(n),
                         opt);
    auto c = gpu.download<float>(dc, n);
    std::vector<float> ref(n);
    for (uint32_t i = 0; i < n; ++i) ref[i] = a[i] + b[i];
    r.max_error = max_abs_diff(c, ref);
    r.correct = r.max_error == 0.0;
    return r;
}

inline RunResult run_strided_copy(Gpu& gpu, const Module& m, uint32_t n, uint32_t stride, uint32_t block = 256) {
    gpu.memory().reset();
    auto in = random_floats((size_t)n * stride, 3);
    uint64_t din = gpu.upload(in), dout = gpu.alloc_zeroed<float>(n);
    RunResult r;
    r.stats = gpu.launch(m.kernel("strided_copy"), {ceil_div(n, block)}, {block},
                         KernelArgs().ptr(din).ptr(dout).u32(n).u32(stride));
    auto out = gpu.download<float>(dout, n);
    std::vector<float> ref(n);
    for (uint32_t i = 0; i < n; ++i) ref[i] = in[(size_t)i * stride];
    r.max_error = max_abs_diff(out, ref);
    r.correct = r.max_error == 0.0;
    return r;
}

inline RunResult run_copy_vec4(Gpu& gpu, const Module& m, uint32_t n, uint32_t block = 256) {
    gpu.memory().reset();
    auto in = random_floats(n, 4);
    uint64_t din = gpu.upload(in), dout = gpu.alloc_zeroed<float>(n);
    RunResult r;
    r.stats = gpu.launch(m.kernel("copy_vec4"), {ceil_div(n / 4, block)}, {block},
                         KernelArgs().ptr(din).ptr(dout).u32(n));
    auto out = gpu.download<float>(dout, n);
    r.max_error = max_abs_diff(out, in);
    r.correct = r.max_error == 0.0;
    return r;
}

// One partial sum per 256-thread CTA.
inline RunResult run_reduce(Gpu& gpu, const Module& m, const std::string& kernel, uint32_t n) {
    const uint32_t block = 256;
    gpu.memory().reset();
    auto in = random_floats(n, 5);
    const uint32_t grid = ceil_div(n, block);
    uint64_t din = gpu.upload(in), dout = gpu.alloc_zeroed<float>(grid);
    RunResult r;
    r.stats = gpu.launch(m.kernel(kernel), {grid}, {block}, KernelArgs().ptr(din).ptr(dout).u32(n));
    auto out = gpu.download<float>(dout, grid);
    r.correct = true;
    for (uint32_t b = 0; b < grid; ++b) {
        double ref = 0;
        for (uint32_t i = b * block; i < std::min(n, (b + 1) * block); ++i) ref += in[i];
        double err = std::fabs(out[b] - ref);
        r.max_error = std::max(r.max_error, err);
        if (err > 1e-3) r.correct = false;  // float summation order differs from the double reference
    }
    return r;
}

inline RunResult run_transpose(Gpu& gpu, const Module& m, const std::string& kernel, uint32_t n) {
    gpu.memory().reset();
    auto in = random_floats((size_t)n * n, 6);
    uint64_t din = gpu.upload(in), dout = gpu.alloc_zeroed<float>((size_t)n * n);
    RunResult r;
    r.stats = gpu.launch(m.kernel(kernel), {n / 32, n / 32}, {32, 8}, KernelArgs().ptr(din).ptr(dout).u32(n));
    auto out = gpu.download<float>(dout, (size_t)n * n);
    std::vector<float> ref((size_t)n * n);
    for (uint32_t y = 0; y < n; ++y)
        for (uint32_t x = 0; x < n; ++x) ref[(size_t)x * n + y] = in[(size_t)y * n + x];
    r.max_error = max_abs_diff(out, ref);
    r.correct = r.max_error == 0.0;
    return r;
}

inline RunResult run_matmul(Gpu& gpu, const Module& m, const std::string& kernel, uint32_t n) {
    gpu.memory().reset();
    auto a = random_floats((size_t)n * n, 7), b = random_floats((size_t)n * n, 8);
    uint64_t da = gpu.upload(a), db = gpu.upload(b), dc = gpu.alloc_zeroed<float>((size_t)n * n);
    RunResult r;
    r.stats = gpu.launch(m.kernel(kernel), {n / 16, n / 16}, {16, 16}, KernelArgs().ptr(da).ptr(db).ptr(dc).u32(n));
    auto c = gpu.download<float>(dc, (size_t)n * n);
    r.correct = true;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            double ref = 0;
            for (uint32_t k = 0; k < n; ++k) ref += (double)a[(size_t)i * n + k] * b[(size_t)k * n + j];
            double err = std::fabs(c[(size_t)i * n + j] - ref);
            r.max_error = std::max(r.max_error, err);
            if (err > 1e-3 * n) r.correct = false;
        }
    return r;
}

// skew = 0: uniform bins; skew = k: only the first 256 >> k bins are used.
inline RunResult run_histogram(Gpu& gpu, const Module& m, const std::string& kernel, uint32_t n, int skew,
                               uint32_t grid, uint32_t block = 256) {
    gpu.memory().reset();
    Lcg rng(9);
    std::vector<uint32_t> in(n);
    for (auto& x : in) x = rng.next() & (255u >> skew);
    uint64_t din = gpu.upload(in), dbins = gpu.alloc_zeroed<uint32_t>(256);
    RunResult r;
    r.stats = gpu.launch(m.kernel(kernel), {grid}, {block}, KernelArgs().ptr(din).ptr(dbins).u32(n));
    auto bins = gpu.download<uint32_t>(dbins, 256);
    std::vector<uint32_t> ref(256, 0);
    for (uint32_t x : in) ref[x & 255]++;
    r.correct = bins == ref;
    return r;
}

}  // namespace harness
