// Cycle-level SIMT GPU model: configuration, occupancy, statistics and the
// public launch API.
#pragma once

#include "memory.h"
#include "ptx.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ptxsim {

enum class SchedPolicy { GreedyThenOldest, LooseRoundRobin };

struct GpuConfig {
    int num_sms = 8;
    int schedulers_per_sm = 4;
    int max_warps_per_sm = 48;
    int max_ctas_per_sm = 16;
    int max_threads_per_cta = 1024;
    int regs_per_sm = 65536;
    int reg_alloc_unit = 256;     // registers are allocated per warp in this granularity
    int smem_per_sm = 64 * 1024;
    int smem_alloc_unit = 256;
    int smem_banks = 32;

    // Latencies in core cycles.
    int lat_alu = 4;
    int lat_sfu = 20;
    int lat_param = 4;
    int lat_shared = 24;
    int lat_l1 = 32;
    int lat_l2 = 190;
    int lat_dram = 400;
    int lat_atomic = 220;

    size_t l1_bytes = 32 * 1024;
    int l1_assoc = 4;
    size_t l2_bytes = 1024 * 1024;
    int l2_assoc = 16;
    double l2_bytes_per_cycle = 256.0;   // aggregate L2 bandwidth
    double dram_bytes_per_cycle = 64.0;  // aggregate DRAM bandwidth

    SchedPolicy policy = SchedPolicy::GreedyThenOldest;
    uint64_t max_cycles = 2'000'000'000ull;  // deadlock guard
};

struct Dim3 {
    uint32_t x = 1, y = 1, z = 1;
    uint32_t count() const { return x * y * z; }
};

// Per-launch resource summary and the SM-level occupancy it allows.
struct Occupancy {
    int warps_per_cta = 0;
    int regs_per_thread = 0;
    int smem_per_cta = 0;
    int ctas_per_sm = 0;
    int active_warps_per_sm = 0;
    double theoretical = 0.0;  // active warps / max warps
    std::string limiter;       // "warps", "ctas", "registers", "shared memory"
};

Occupancy compute_occupancy(const GpuConfig& cfg, int threads_per_cta, int regs_per_thread, int smem_per_cta);

enum StallReason : int {
    kIssued = 0,
    kStallLongScoreboard,   // waiting on a global-memory / atomic result
    kStallShortScoreboard,  // waiting on a shared-memory / SFU / shuffle result
    kStallWait,             // fixed-latency ALU dependency
    kStallBarrier,          // waiting at bar.sync
    kStallMemThrottle,      // load/store unit busy (bank conflict replays, many transactions)
    kStallIdle,             // scheduler has no resident warps
    kNumStallReasons
};

const char* stall_reason_name(int r);

struct KernelStats {
    std::string kernel;
    uint64_t cycles = 0;
    uint64_t warp_instructions = 0;
    uint64_t thread_instructions = 0;  // sum of active lanes over issued instructions
    uint64_t branches = 0;
    uint64_t divergent_branches = 0;
    uint64_t barriers = 0;

    uint64_t gld_requests = 0, gld_sectors = 0;
    uint64_t gst_requests = 0, gst_sectors = 0;
    uint64_t l1_hits = 0, l1_misses = 0;  // sectors
    uint64_t l2_hits = 0, l2_misses = 0;  // sectors
    uint64_t dram_read_bytes = 0, dram_write_bytes = 0;
    uint64_t shared_requests = 0, shared_wavefronts = 0;
    uint64_t atomic_requests = 0, atomic_serialized = 0;

    Occupancy occupancy;
    double achieved_occupancy = 0.0;
    std::array<uint64_t, kNumStallReasons> sched_cycles{};  // scheduler-cycles by state

    double ipc() const { return cycles ? (double)warp_instructions / cycles : 0.0; }
    double simd_efficiency() const {
        return warp_instructions ? (double)thread_instructions / (32.0 * warp_instructions) : 0.0;
    }
    double gld_sectors_per_request() const { return gld_requests ? (double)gld_sectors / gld_requests : 0.0; }
    double gst_sectors_per_request() const { return gst_requests ? (double)gst_sectors / gst_requests : 0.0; }
    double l1_hit_rate() const { return l1_hits + l1_misses ? (double)l1_hits / (l1_hits + l1_misses) : 0.0; }
    double l2_hit_rate() const { return l2_hits + l2_misses ? (double)l2_hits / (l2_hits + l2_misses) : 0.0; }
    uint64_t bank_conflicts() const { return shared_wavefronts - shared_requests; }
    double stall_fraction(int r) const;
    std::string report() const;
};

struct LaunchOptions {
    int dynamic_shared_bytes = 0;
    int regs_per_thread = 0;   // 0 = estimate from the kernel's .reg declarations
};

// Kernel arguments in declaration order. Sizes are checked against the
// kernel's .param list at launch.
class KernelArgs {
public:
    KernelArgs& ptr(uint64_t v) { return push(v, 8); }
    KernelArgs& u64(uint64_t v) { return push(v, 8); }
    KernelArgs& u32(uint32_t v) { return push(v, 4); }
    KernelArgs& s32(int32_t v) { return push((uint32_t)v, 4); }
    KernelArgs& f32(float v) {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return push(b, 4);
    }
    struct Arg { uint64_t bits; int size; };
    const std::vector<Arg>& args() const { return args_; }

private:
    KernelArgs& push(uint64_t v, int size) { args_.push_back({v, size}); return *this; }
    std::vector<Arg> args_;
};

struct SimError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Gpu {
public:
    explicit Gpu(GpuConfig cfg = GpuConfig());
    ~Gpu();

    const GpuConfig& config() const { return cfg_; }
    GlobalMemory& memory() { return mem_; }

    template <class T>
    uint64_t upload(const std::vector<T>& v) {
        uint64_t a = mem_.alloc(v.size() * sizeof(T));
        mem_.write(a, v.data(), v.size() * sizeof(T));
        return a;
    }
    template <class T>
    uint64_t alloc_zeroed(size_t n) { return mem_.alloc(n * sizeof(T)); }
    template <class T>
    std::vector<T> download(uint64_t addr, size_t n) const {
        std::vector<T> v(n);
        mem_.read(addr, v.data(), n * sizeof(T));
        return v;
    }

    // Every launch starts with cold L1/L2 caches (like a profiler replay).
    KernelStats launch(const Kernel& k, Dim3 grid, Dim3 block, const KernelArgs& args,
                       const LaunchOptions& opt = LaunchOptions());

private:
    GpuConfig cfg_;
    GlobalMemory mem_;
};

}  // namespace ptxsim
