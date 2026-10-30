// End-to-end tests of the repository kernels: functional correctness against
// CPU references, plus the architectural effect each kernel variant exists to
// show (coalescing, bank conflicts, divergence, data reuse, atomics).
#include "kernel_harness.h"
#include "test.h"

using namespace harness;

static const Module& kernels() {
    static Module m = load_kernels(testing::kernel_dir());
    return m;
}

static GpuConfig small_gpu() {
    GpuConfig c;
    c.num_sms = 4;
    return c;
}

TEST(vecadd_correct_and_fully_coalesced) {
    Gpu gpu(small_gpu());
    RunResult r = run_vecadd(gpu, kernels(), 10000);  // not a multiple of the block size
    CHECK(r.correct);
    CHECK_NEAR(r.stats.gld_sectors_per_request(), 4.0, 0.05);
    CHECK_EQ(r.stats.dram_read_bytes, (uint64_t)2 * 10000 * 4);  // each input byte read from DRAM exactly once
}

TEST(strided_copy_sectors_per_request_grow_with_stride) {
    Gpu gpu(small_gpu());
    double prev = 0;
    for (uint32_t stride : {1u, 2u, 4u, 8u, 16u}) {
        RunResult r = run_strided_copy(gpu, kernels(), 8192, stride);
        CHECK(r.correct);
        double spr = r.stats.gld_sectors_per_request();
        CHECK_NEAR(spr, std::min(32.0, 4.0 * stride), 0.01);
        CHECK(spr >= prev);  // saturates at one sector per lane from stride 8 on
        prev = spr;
    }
}

TEST(vector_loads_issue_fewer_instructions) {
    Gpu gpu(small_gpu());
    RunResult scalar = run_strided_copy(gpu, kernels(), 1 << 16, 1);
    RunResult vec = run_copy_vec4(gpu, kernels(), 1 << 16);
    CHECK(scalar.correct && vec.correct);
    CHECK_EQ(vec.stats.gld_requests * 4, scalar.stats.gld_requests);
    CHECK_NEAR(vec.stats.gld_sectors_per_request(), 16.0, 0.01);
    CHECK(vec.stats.warp_instructions < scalar.stats.warp_instructions / 2);
}

TEST(reductions_are_correct) {
    Gpu gpu(small_gpu());
    for (const char* k : {"reduce_interleaved", "reduce_strided_index", "reduce_sequential", "reduce_warp_shuffle"}) {
        RunResult r = run_reduce(gpu, kernels(), k, 5000);  // partial last block
        CHECK(r.correct);
    }
}

TEST(reduction_variants_show_divergence_and_bank_conflicts) {
    Gpu gpu(small_gpu());
    const uint32_t n = 1 << 14;
    RunResult inter = run_reduce(gpu, kernels(), "reduce_interleaved", n);
    RunResult strided = run_reduce(gpu, kernels(), "reduce_strided_index", n);
    RunResult seq = run_reduce(gpu, kernels(), "reduce_sequential", n);
    RunResult shfl = run_reduce(gpu, kernels(), "reduce_warp_shuffle", n);
    // tid % (2s) leaves most lanes idle inside every warp.
    CHECK(inter.stats.divergent_branches > 0);
    CHECK(inter.stats.simd_efficiency() < seq.stats.simd_efficiency());
    // 2*s*tid indexing: no intra-warp divergence in the early steps, but strided shared accesses conflict.
    CHECK(strided.stats.bank_conflicts() > 0);
    CHECK_EQ(seq.stats.bank_conflicts(), (uint64_t)0);
    CHECK(strided.stats.cycles < inter.stats.cycles);
    CHECK(seq.stats.cycles < strided.stats.cycles);
    // Shuffles replace most shared-memory traffic and barriers.
    CHECK(shfl.stats.shared_requests < seq.stats.shared_requests / 4);
    CHECK(shfl.stats.barriers < seq.stats.barriers);
    CHECK(shfl.stats.cycles < seq.stats.cycles);
}

TEST(transpose_variants_correct_and_padding_removes_conflicts) {
    Gpu gpu(small_gpu());
    RunResult naive = run_transpose(gpu, kernels(), "transpose_naive", 128);
    RunResult smem = run_transpose(gpu, kernels(), "transpose_smem", 128);
    RunResult pad = run_transpose(gpu, kernels(), "transpose_smem_padded", 128);
    CHECK(naive.correct && smem.correct && pad.correct);
    CHECK(naive.stats.gst_sectors_per_request() > 30);   // column writes: one sector per lane
    CHECK_NEAR(smem.stats.gst_sectors_per_request(), 4.0, 0.01);
    // Reading a column of a 32x32 tile: all 32 lanes hit one bank.
    CHECK_EQ(smem.stats.shared_wavefronts, smem.stats.shared_requests / 2 * (1 + 32));
    CHECK_EQ(pad.stats.bank_conflicts(), (uint64_t)0);
    CHECK(pad.stats.cycles < smem.stats.cycles);
}

TEST(matmul_variants_correct_and_tiling_cuts_global_traffic) {
    Gpu gpu(small_gpu());
    RunResult naive = run_matmul(gpu, kernels(), "matmul_naive", 64);
    RunResult tiled = run_matmul(gpu, kernels(), "matmul_tiled", 64);
    CHECK(naive.correct && tiled.correct);
    // naive: 2 loads per k per thread; tiled: 2 loads per 16-wide tile per thread
    CHECK_EQ(naive.stats.gld_requests, tiled.stats.gld_requests * 16);
    CHECK_EQ(tiled.stats.bank_conflicts(), (uint64_t)0);
    CHECK(tiled.stats.cycles < naive.stats.cycles);
}

TEST(histograms_correct_and_privatization_cuts_global_atomics) {
    Gpu gpu(small_gpu());
    for (int skew : {0, 6}) {
        RunResult g = run_histogram(gpu, kernels(), "histogram_global", 1 << 14, skew, 16);
        RunResult s = run_histogram(gpu, kernels(), "histogram_shared", 1 << 14, skew, 16);
        CHECK(g.correct && s.correct);
        CHECK(s.stats.cycles < g.stats.cycles);
    }
    RunResult uniform = run_histogram(gpu, kernels(), "histogram_global", 1 << 14, 0, 16);
    RunResult skewed = run_histogram(gpu, kernels(), "histogram_global", 1 << 14, 6, 16);  // 4 distinct bins
    CHECK(skewed.stats.atomic_serialized > uniform.stats.atomic_serialized);
    CHECK(skewed.stats.cycles > uniform.stats.cycles);
}

TEST(occupancy_limit_reduces_latency_hiding) {
    // Same memory-bound kernel; dynamic shared memory caps resident CTAs per SM.
    Gpu gpu(small_gpu());
    LaunchOptions full, limited;
    limited.dynamic_shared_bytes = 60 * 1024;  // one CTA per SM
    RunResult a = run_vecadd(gpu, kernels(), 1 << 16, 128, full);
    RunResult b = run_vecadd(gpu, kernels(), 1 << 16, 128, limited);
    CHECK(a.correct && b.correct);
    CHECK_EQ(b.stats.occupancy.ctas_per_sm, 1);
    CHECK(b.stats.achieved_occupancy < a.stats.achieved_occupancy);
    CHECK(b.stats.cycles > a.stats.cycles);
}

TEST(scheduler_policies_give_identical_results) {
    GpuConfig gto = small_gpu(), lrr = small_gpu();
    lrr.policy = SchedPolicy::LooseRoundRobin;
    Gpu g1(gto), g2(lrr);
    RunResult a = run_matmul(g1, kernels(), "matmul_tiled", 64);
    RunResult b = run_matmul(g2, kernels(), "matmul_tiled", 64);
    CHECK(a.correct && b.correct);
    CHECK_EQ(a.stats.warp_instructions, b.stats.warp_instructions);
}

TEST(simulation_is_deterministic) {
    Gpu gpu(small_gpu());
    RunResult a = run_transpose(gpu, kernels(), "transpose_smem", 64);
    RunResult b = run_transpose(gpu, kernels(), "transpose_smem", 64);
    CHECK_EQ(a.stats.cycles, b.stats.cycles);
    CHECK_EQ(a.stats.l2_hits, b.stats.l2_hits);
}

TEST(scheduler_cycle_accounting_covers_every_cycle) {
    GpuConfig cfg = small_gpu();
    Gpu gpu(cfg);
    for (const char* k : {"matmul_naive", "matmul_tiled"}) {
        RunResult r = run_matmul(gpu, kernels(), k, 64);
        uint64_t total = 0;
        for (auto c : r.stats.sched_cycles) total += c;
        CHECK_EQ(total, r.stats.cycles * cfg.num_sms * cfg.schedulers_per_sm);
        CHECK_EQ(r.stats.sched_cycles[kIssued], r.stats.warp_instructions);
    }
}
