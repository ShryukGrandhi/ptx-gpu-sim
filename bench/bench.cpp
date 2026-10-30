// Benchmark: runs every kernel variant on the simulated GPU, verifies the
// results against CPU references and prints the architectural metrics as
// markdown tables. Also reports how fast the simulator itself runs.
//
// Usage: ptxsim_bench <kernel_dir> [--quick]
#include "kernel_harness.h"

#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace harness;

namespace {

struct Totals {
    uint64_t warp_instructions = 0;
    uint64_t thread_instructions = 0;
    uint64_t cycles = 0;
    double seconds = 0;
    int runs = 0;
    int failures = 0;
} totals;

// Runs one simulation, timing the host and checking correctness.
RunResult measure(const std::function<RunResult()>& fn, const std::string& what) {
    auto t0 = std::chrono::steady_clock::now();
    RunResult r = fn();
    auto t1 = std::chrono::steady_clock::now();
    totals.seconds += std::chrono::duration<double>(t1 - t0).count();
    totals.warp_instructions += r.stats.warp_instructions;
    totals.thread_instructions += r.stats.thread_instructions;
    totals.cycles += r.stats.cycles;
    totals.runs++;
    if (!r.correct) {
        totals.failures++;
        std::cerr << "WRONG RESULT: " << what << " (max error " << r.max_error << ")\n";
    }
    return r;
}

std::string fmt(double v, int prec = 2) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}
std::string pct(double v) { return fmt(100.0 * v, 1) + "%"; }
std::string num(uint64_t v) { return std::to_string(v); }

void table(const std::vector<std::string>& head, const std::vector<std::vector<std::string>>& rows) {
    std::cout << "|";
    for (auto& h : head) std::cout << " " << h << " |";
    std::cout << "\n|";
    for (size_t i = 0; i < head.size(); ++i) std::cout << (i == 0 ? "---|" : "---:|");
    std::cout << "\n";
    for (auto& r : rows) {
        std::cout << "|";
        for (auto& c : r) std::cout << " " << c << " |";
        std::cout << "\n";
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "kernels";
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick") quick = true;
        else dir = a;
    }
    const Module m = load_kernels(dir);
    GpuConfig cfg;
    Gpu gpu(cfg);

    std::cout << "# ptxsim benchmark" << (quick ? " (--quick sizes)" : "") << "\n\n";
    std::cout << "Simulated GPU: " << cfg.num_sms << " SMs x " << cfg.schedulers_per_sm << " schedulers, "
              << cfg.max_warps_per_sm << " warps/SM, " << cfg.l1_bytes / 1024 << " KiB L1/SM, "
              << cfg.l2_bytes / 1024 << " KiB L2, DRAM " << cfg.dram_bytes_per_cycle << " B/cycle, "
              << "latencies L1/L2/DRAM " << cfg.lat_l1 << "/" << cfg.lat_l2 << "/" << cfg.lat_dram << " cycles.\n\n";

    // 1. Coalescing ---------------------------------------------------------
    {
        const uint32_t n = quick ? 1 << 15 : 1 << 18;
        std::cout << "## Global memory coalescing: strided_copy, n = " << n << "\n\n";
        std::vector<std::vector<std::string>> rows;
        uint64_t base = 0;
        for (uint32_t stride : {1u, 2u, 4u, 8u, 32u}) {
            RunResult r = measure([&] { return run_strided_copy(gpu, m, n, stride); }, "strided_copy");
            if (stride == 1) base = r.stats.cycles;
            double useful = 2.0 * n * 4 / r.stats.cycles;
            rows.push_back({num(stride), fmt(r.stats.gld_sectors_per_request()), num(r.stats.dram_read_bytes),
                            num(r.stats.cycles), fmt(useful), fmt((double)r.stats.cycles / base) + "x"});
        }
        table({"stride", "sectors/request", "DRAM bytes read", "cycles", "useful B/cycle (load + store)", "slowdown"}, rows);
    }

    // 2. Vector loads ---------------------------------------------------------
    {
        const uint32_t n = quick ? 1 << 15 : 1 << 18;
        std::cout << "## Vectorized access: scalar vs float4 copy, n = " << n << "\n\n";
        RunResult s = measure([&] { return run_strided_copy(gpu, m, n, 1); }, "copy scalar");
        RunResult v = measure([&] { return run_copy_vec4(gpu, m, n); }, "copy_vec4");
        table({"kernel", "warp instructions", "load requests", "sectors/request", "cycles"},
              {{"strided_copy (stride 1)", num(s.stats.warp_instructions), num(s.stats.gld_requests),
                fmt(s.stats.gld_sectors_per_request()), num(s.stats.cycles)},
               {"copy_vec4", num(v.stats.warp_instructions), num(v.stats.gld_requests),
                fmt(v.stats.gld_sectors_per_request()), num(v.stats.cycles)}});
    }

    // 3. Reductions -----------------------------------------------------------
    {
        const uint32_t n = quick ? 1 << 16 : 1 << 20;
        std::cout << "## Block reduction variants, n = " << n << ", 256 threads/CTA\n\n";
        std::vector<std::vector<std::string>> rows;
        uint64_t base = 0;
        for (const char* k : {"reduce_interleaved", "reduce_strided_index", "reduce_sequential", "reduce_warp_shuffle"}) {
            RunResult r = measure([&] { return run_reduce(gpu, m, k, n); }, k);
            if (!base) base = r.stats.cycles;
            rows.push_back({k, num(r.stats.cycles), fmt((double)base / r.stats.cycles) + "x",
                            pct(r.stats.simd_efficiency()), num(r.stats.divergent_branches),
                            num(r.stats.shared_requests), num(r.stats.bank_conflicts()), num(r.stats.warp_instructions)});
        }
        table({"kernel", "cycles", "speedup", "SIMD efficiency", "divergent branches", "shared requests",
               "bank-conflict wavefronts", "warp instructions"},
              rows);
    }

    // 4. Transpose -------------------------------------------------------------
    {
        const uint32_t n = quick ? 128 : 512;
        std::cout << "## Matrix transpose, " << n << " x " << n << " floats\n\n";
        std::vector<std::vector<std::string>> rows;
        uint64_t base = 0;
        for (const char* k : {"transpose_naive", "transpose_smem", "transpose_smem_padded"}) {
            RunResult r = measure([&] { return run_transpose(gpu, m, k, n); }, k);
            if (!base) base = r.stats.cycles;
            rows.push_back({k, num(r.stats.cycles), fmt((double)base / r.stats.cycles) + "x",
                            fmt(r.stats.gld_sectors_per_request()), fmt(r.stats.gst_sectors_per_request()),
                            fmt(r.stats.shared_requests ? (double)r.stats.shared_wavefronts / r.stats.shared_requests : 0.0),
                            pct(r.stats.stall_fraction(kStallMemThrottle))});
        }
        table({"kernel", "cycles", "speedup", "load sectors/req", "store sectors/req", "shared wavefronts/req",
               "mem_throttle stalls"},
              rows);
    }

    // 5. Matmul -----------------------------------------------------------------
    {
        const uint32_t n = quick ? 64 : 256;
        std::cout << "## SGEMM, " << n << " x " << n << ", 16x16 CTAs\n\n";
        std::vector<std::vector<std::string>> rows;
        uint64_t base = 0;
        for (const char* k : {"matmul_naive", "matmul_tiled"}) {
            RunResult r = measure([&] { return run_matmul(gpu, m, k, n); }, k);
            if (!base) base = r.stats.cycles;
            double flops_per_cycle = 2.0 * n * n * n / r.stats.cycles;
            rows.push_back({k, num(r.stats.cycles), fmt((double)base / r.stats.cycles) + "x",
                            fmt(flops_per_cycle), num(r.stats.gld_requests), pct(r.stats.l1_hit_rate()),
                            num(r.stats.dram_read_bytes), fmt(r.stats.ipc()),
                            pct(r.stats.stall_fraction(kStallLongScoreboard))});
        }
        table({"kernel", "cycles", "speedup", "FLOP/cycle", "global load requests", "L1 hit rate", "DRAM bytes read",
               "IPC", "long_scoreboard stalls"},
              rows);
    }

    // 6. Histogram ---------------------------------------------------------------
    {
        const uint32_t n = quick ? 1 << 15 : 1 << 18;
        std::cout << "## Histogram (256 bins), n = " << n << ", 64 CTAs x 256 threads\n\n";
        std::vector<std::vector<std::string>> rows;
        for (int skew : {0, 6}) {
            for (const char* k : {"histogram_global", "histogram_shared"}) {
                RunResult r = measure([&] { return run_histogram(gpu, m, k, n, skew, 64); }, k);
                rows.push_back({k, skew ? "4 bins (skewed)" : "256 bins (uniform)", num(r.stats.cycles),
                                num(r.stats.atomic_requests), num(r.stats.atomic_serialized)});
            }
        }
        table({"kernel", "input", "cycles", "warp atomic requests", "serialized lane updates"}, rows);
    }

    // 7. Occupancy -----------------------------------------------------------------
    {
        const uint32_t n = quick ? 1 << 15 : 1 << 18;
        std::cout << "## Occupancy vs latency hiding: vecadd, n = " << n
                  << ", 128 threads/CTA, CTAs/SM capped with dynamic shared memory\n\n";
        std::vector<std::vector<std::string>> rows;
        for (int ctas : {1, 2, 4, 8, 12}) {
            LaunchOptions opt;
            opt.dynamic_shared_bytes = cfg.smem_per_sm / ctas - 256;
            RunResult r = measure([&] { return run_vecadd(gpu, m, n, 128, opt); }, "vecadd occupancy");
            rows.push_back({num(r.stats.occupancy.ctas_per_sm), pct(r.stats.occupancy.theoretical),
                            pct(r.stats.achieved_occupancy), num(r.stats.cycles),
                            fmt(3.0 * n * 4 / r.stats.cycles), pct(r.stats.stall_fraction(kStallLongScoreboard))});
        }
        table({"CTAs/SM", "theoretical occupancy", "achieved occupancy", "cycles", "kernel B/cycle (2 loads + 1 store)",
               "long_scoreboard stalls"},
              rows);
    }

    // 8. Scheduler policy --------------------------------------------------------------
    {
        const uint32_t mm = quick ? 64 : 256, tn = quick ? 128 : 512, vn = quick ? 1 << 15 : 1 << 18;
        std::cout << "## Warp scheduling policy: greedy-then-oldest vs loose round-robin\n\n";
        GpuConfig lrr_cfg = cfg;
        lrr_cfg.policy = SchedPolicy::LooseRoundRobin;
        Gpu lrr(lrr_cfg);
        std::vector<std::vector<std::string>> rows;
        auto add = [&](const std::string& name, const std::function<RunResult(Gpu&)>& f) {
            RunResult a = measure([&] { return f(gpu); }, name + " GTO");
            RunResult b = measure([&] { return f(lrr); }, name + " LRR");
            rows.push_back({name, num(a.stats.cycles), num(b.stats.cycles), pct(a.stats.l1_hit_rate()),
                            pct(b.stats.l1_hit_rate())});
        };
        add("matmul_naive " + std::to_string(mm), [&](Gpu& g) { return run_matmul(g, m, "matmul_naive", mm); });
        add("matmul_tiled " + std::to_string(mm), [&](Gpu& g) { return run_matmul(g, m, "matmul_tiled", mm); });
        add("transpose_smem_padded " + std::to_string(tn),
            [&](Gpu& g) { return run_transpose(g, m, "transpose_smem_padded", tn); });
        add("vecadd " + std::to_string(vn), [&](Gpu& g) { return run_vecadd(g, m, vn); });
        table({"workload", "GTO cycles", "LRR cycles", "GTO L1 hit rate", "LRR L1 hit rate"}, rows);
    }

    // 9. Simulator speed ------------------------------------------------------------------
    std::cout << "## Simulator throughput (host)\n\n";
    table({"simulations", "simulated cycles", "warp instructions", "thread instructions", "host seconds",
           "warp instr / s", "thread instr / s"},
          {{num(totals.runs), num(totals.cycles), num(totals.warp_instructions), num(totals.thread_instructions),
            fmt(totals.seconds), fmt(totals.warp_instructions / totals.seconds / 1e6) + " M",
            fmt(totals.thread_instructions / totals.seconds / 1e6) + " M"}});
    std::cout << "Host time includes input generation and CPU reference checks.\n";
    std::cout << "All results verified against CPU references: " << (totals.failures ? "NO" : "yes") << "\n";
    return totals.failures ? 1 : 0;
}
