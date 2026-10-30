// Execution-engine tests on small inline PTX kernels: SIMT divergence and
// reconvergence, barriers, shuffles, atomics, arithmetic semantics, faults
// and basic timing properties.
#include "gpu.h"
#include "ptx.h"
#include "test.h"

#include <algorithm>
#include <cstring>

using namespace ptxsim;

// Wraps a body in a kernel that writes %r1 to out[global thread id].
// Available: %r0 = tid.x, %r28 = global id, %r30 = argument, shared array `sm`.
static Kernel wrap(const std::string& body) {
    std::string src = R"(
.visible .entry k(.param .u64 k_out, .param .u32 k_arg)
{
	.reg .pred %p<8>;
	.reg .b32 %r<32>;
	.reg .f32 %f<8>;
	.reg .b64 %rd<8>;
	.shared .align 4 .b8 sm[4096];
	ld.param.u64 %rd1, [k_out];
	cvta.to.global.u64 %rd1, %rd1;
	ld.param.u32 %r30, [k_arg];
	mov.u32 %r0, %tid.x;
	mov.u32 %r31, %ctaid.x;
	mov.u32 %r29, %ntid.x;
	mad.lo.s32 %r28, %r31, %r29, %r0;
	mul.wide.u32 %rd2, %r28, 4;
	add.s64 %rd3, %rd1, %rd2;
)" + body + R"(
	st.global.u32 [%rd3], %r1;
	ret;
}
)";
    return parse_ptx(src).kernel("k");
}

struct Out {
    std::vector<uint32_t> v;
    KernelStats st;
};

static Out run(const Kernel& k, uint32_t threads, uint32_t block, uint32_t arg = 0, GpuConfig cfg = GpuConfig()) {
    Gpu gpu(cfg);
    uint64_t out = gpu.upload(std::vector<uint32_t>(threads, 0xdeadbeefu));
    Out o;
    o.st = gpu.launch(k, {threads / block}, {block}, KernelArgs().ptr(out).u32(arg));
    o.v = gpu.download<uint32_t>(out, threads);
    return o;
}

TEST(divergent_if_else_reconverges) {
    Kernel k = wrap(R"(
	setp.lt.u32 %p1, %r0, 10;
	@%p1 bra $L__then;
	add.s32 %r1, %r0, 100;
	bra.uni $L__join;
$L__then:
	shl.b32 %r1, %r0, 1;
$L__join:
	add.s32 %r1, %r1, 1;
)");
    Out o = run(k, 32, 32);
    for (uint32_t t = 0; t < 32; ++t) CHECK_EQ(o.v[t], t < 10 ? 2 * t + 1 : t + 101);
    CHECK_EQ(o.st.divergent_branches, (uint64_t)1);
    CHECK(o.st.simd_efficiency() < 1.0);
}

TEST(nested_divergence) {
    Kernel k = wrap(R"(
	and.b32 %r2, %r0, 1;
	setp.eq.s32 %p1, %r2, 0;
	@%p1 bra $L__even;
	and.b32 %r3, %r0, 3;
	setp.eq.s32 %p2, %r3, 1;
	@%p2 bra $L__one;
	mov.u32 %r1, 3;
	bra.uni $L__odd_join;
$L__one:
	mov.u32 %r1, 1;
$L__odd_join:
	add.s32 %r1, %r1, 10;
	bra.uni $L__join;
$L__even:
	mov.u32 %r1, 0;
$L__join:
	add.s32 %r1, %r1, 1000;
)");
    Out o = run(k, 64, 64);
    for (uint32_t t = 0; t < 64; ++t) {
        uint32_t want = (t % 2 == 0) ? 1000 : (t % 4 == 1 ? 1011 : 1013);
        CHECK_EQ(o.v[t], want);
    }
    CHECK_EQ(o.st.divergent_branches, (uint64_t)4);  // two per warp
}

TEST(per_lane_loop_trip_counts) {
    Kernel k = wrap(R"(
	mov.u32 %r1, 0;
	mov.u32 %r2, 0;
$L__loop:
	setp.ge.u32 %p1, %r2, %r0;
	@%p1 bra $L__done;
	add.s32 %r1, %r1, %r2;
	add.s32 %r2, %r2, 1;
	bra.uni $L__loop;
$L__done:
)");
    Out o = run(k, 64, 32);
    for (uint32_t t = 0; t < 64; ++t) {
        uint32_t lane = t % 32;
        CHECK_EQ(o.v[t], lane * (lane - 1) / 2);  // lane 0: 0 * (2^32 - 1) / 2 == 0
    }
}

TEST(early_return_lanes_do_not_execute_later_code) {
    Kernel k = wrap(R"(
	setp.ge.u32 %p1, %r0, 5;
	@%p1 ret;
	mov.u32 %r1, 7;
)");
    Out o = run(k, 32, 32);
    for (uint32_t t = 0; t < 32; ++t) CHECK_EQ(o.v[t], t < 5 ? 7u : 0xdeadbeefu);
}

TEST(barrier_orders_shared_memory_between_warps) {
    Kernel k = wrap(R"(
	mov.u32 %r2, sm;
	shl.b32 %r3, %r0, 2;
	add.s32 %r4, %r2, %r3;
	mul.lo.s32 %r5, %r0, 3;
	st.shared.u32 [%r4], %r5;
	bar.sync 0;
	sub.s32 %r6, 127, %r0;
	shl.b32 %r7, %r6, 2;
	add.s32 %r8, %r2, %r7;
	ld.shared.u32 %r1, [%r8];
)");
    Out o = run(k, 256, 128);
    for (uint32_t t = 0; t < 256; ++t) CHECK_EQ(o.v[t], (127 - t % 128) * 3);
    CHECK_EQ(o.st.barriers, (uint64_t)2);
}

TEST(barrier_completes_when_other_warps_already_exited) {
    Kernel k = wrap(R"(
	setp.ge.u32 %p1, %r0, 32;
	@%p1 ret;
	bar.sync 0;
	mov.u32 %r1, 1;
)");
    Out o = run(k, 64, 64);
    for (uint32_t t = 0; t < 32; ++t) CHECK_EQ(o.v[t], 1u);
}

TEST(shuffle_modes) {
    // %r30 selects the mode so one kernel covers all four.
    Kernel k = wrap(R"(
	mul.lo.s32 %r2, %r0, 10;
	setp.eq.s32 %p1, %r30, 0;
	@%p1 bra $L__down;
	setp.eq.s32 %p1, %r30, 1;
	@%p1 bra $L__up;
	setp.eq.s32 %p1, %r30, 2;
	@%p1 bra $L__bfly;
	shfl.sync.idx.b32 %r1, %r2, 5, 31, -1;
	bra.uni $L__end;
$L__down:
	shfl.sync.down.b32 %r1|%p2, %r2, 3, 31, -1;
	bra.uni $L__end;
$L__up:
	shfl.sync.up.b32 %r1, %r2, 2, 0, -1;
	bra.uni $L__end;
$L__bfly:
	shfl.sync.bfly.b32 %r1, %r2, 1, 31, -1;
$L__end:
)");
    Out down = run(k, 32, 32, 0), up = run(k, 32, 32, 1), bfly = run(k, 32, 32, 2), idx = run(k, 32, 32, 3);
    for (uint32_t l = 0; l < 32; ++l) {
        CHECK_EQ(down.v[l], (l + 3 <= 31 ? l + 3 : l) * 10);
        CHECK_EQ(up.v[l], (l >= 2 ? l - 2 : l) * 10);
        CHECK_EQ(bfly.v[l], (l ^ 1) * 10);
        CHECK_EQ(idx.v[l], 50u);
    }
}

TEST(shuffle_with_segment_width_8) {
    // c = ((32 - 8) << 8) | 31 : __shfl_down_sync(mask, v, 1, 8)
    Kernel k = wrap(R"(
	shfl.sync.down.b32 %r1, %r0, 1, 6175, -1;
)");
    Out o = run(k, 32, 32);
    for (uint32_t l = 0; l < 32; ++l) CHECK_EQ(o.v[l], (l % 8 == 7) ? l : l + 1);
}

TEST(global_atomics_are_serialized_and_complete) {
    const char* src = R"(
.visible .entry count(.param .u64 count_ctr, .param .u64 count_old)
{
	.reg .b32 %r<6>;
	.reg .b64 %rd<6>;
	ld.param.u64 %rd1, [count_ctr];
	ld.param.u64 %rd2, [count_old];
	mov.u32 %r1, %ctaid.x;
	mov.u32 %r2, %ntid.x;
	mov.u32 %r3, %tid.x;
	mad.lo.s32 %r4, %r1, %r2, %r3;
	atom.global.add.u32 %r5, [%rd1], 1;
	mul.wide.u32 %rd3, %r4, 4;
	add.s64 %rd4, %rd2, %rd3;
	st.global.u32 [%rd4], %r5;
	ret;
}
)";
    Kernel k = parse_ptx(src).kernel("count");
    Gpu gpu;
    uint64_t ctr = gpu.alloc_zeroed<uint32_t>(1), old = gpu.alloc_zeroed<uint32_t>(512);
    KernelStats st = gpu.launch(k, {4}, {128}, KernelArgs().ptr(ctr).ptr(old));
    CHECK_EQ(gpu.download<uint32_t>(ctr, 1)[0], 512u);
    auto olds = gpu.download<uint32_t>(old, 512);
    std::sort(olds.begin(), olds.end());
    for (uint32_t i = 0; i < 512; ++i) CHECK_EQ(olds[i], i);  // every old value is unique
    CHECK_EQ(st.atomic_requests, (uint64_t)16);
    CHECK_EQ(st.atomic_serialized, (uint64_t)(512 - 16));
}

TEST(integer_and_float_semantics) {
    const char* src = R"(
.visible .entry alu(.param .u64 alu_out)
{
	.reg .pred %p<3>;
	.reg .b32 %r<16>;
	.reg .f32 %f<6>;
	.reg .b64 %rd<6>;
	ld.param.u64 %rd1, [alu_out];
	mov.u32 %r1, -7;
	mov.u32 %r2, 2;
	shr.s32 %r3, %r1, 1;          // arithmetic: -4
	st.global.u32 [%rd1+0], %r3;
	shr.u32 %r3, %r1, 28;         // logical: 15
	st.global.u32 [%rd1+4], %r3;
	rem.s32 %r3, %r1, %r2;        // -1
	st.global.u32 [%rd1+8], %r3;
	div.s32 %r3, %r1, %r2;        // -3
	st.global.u32 [%rd1+12], %r3;
	mov.u32 %r4, 0x80000000;
	mul.hi.u32 %r3, %r4, 6;       // 3
	st.global.u32 [%rd1+16], %r3;
	mul.wide.s32 %rd2, %r1, 1000000;
	st.global.u64 [%rd1+24], %rd2; // -7000000
	min.s32 %r3, %r1, %r2;        // -7
	st.global.u32 [%rd1+32], %r3;
	min.u32 %r3, %r1, %r2;        // 2
	st.global.u32 [%rd1+36], %r3;
	mov.f32 %f1, 0fC0300000;      // -2.75
	cvt.rzi.s32.f32 %r3, %f1;     // -2
	st.global.u32 [%rd1+40], %r3;
	cvt.rn.f32.s32 %f2, %r1;      // -7.0
	st.global.f32 [%rd1+44], %f2;
	div.rn.f32 %f3, %f2, %f1;     // 2.545454...
	st.global.f32 [%rd1+48], %f3;
	setp.lt.f32 %p1, %f1, %f2;    // -2.75 < -7 : false
	selp.b32 %r3, 11, 22, %p1;
	st.global.u32 [%rd1+52], %r3;
	shl.b32 %r3, %r2, 33;         // shift >= width -> 0
	st.global.u32 [%rd1+56], %r3;
	cvt.u64.u32 %rd3, %r1;        // zero-extend
	st.global.u64 [%rd1+64], %rd3;
	cvt.s64.s32 %rd3, %r1;        // sign-extend
	st.global.u64 [%rd1+72], %rd3;
	ret;
}
)";
    Kernel k = parse_ptx(src).kernel("alu");
    Gpu gpu;
    uint64_t out = gpu.alloc_zeroed<uint8_t>(80);
    gpu.launch(k, {1}, {1}, KernelArgs().ptr(out));
    auto b = gpu.download<uint8_t>(out, 80);
    auto u32 = [&](int off) { uint32_t v; std::memcpy(&v, &b[off], 4); return (int32_t)v; };
    auto u64 = [&](int off) { uint64_t v; std::memcpy(&v, &b[off], 8); return v; };
    auto f32 = [&](int off) { float v; std::memcpy(&v, &b[off], 4); return v; };
    CHECK_EQ(u32(0), -4);
    CHECK_EQ(u32(4), 15);
    CHECK_EQ(u32(8), -1);
    CHECK_EQ(u32(12), -3);
    CHECK_EQ(u32(16), 3);
    CHECK_EQ((int64_t)u64(24), (int64_t)-7000000);
    CHECK_EQ(u32(32), -7);
    CHECK_EQ(u32(36), 2);
    CHECK_EQ(u32(40), -2);
    CHECK_NEAR(f32(44), -7.0, 0);
    CHECK_NEAR(f32(48), -7.0f / -2.75f, 1e-6);
    CHECK_EQ(u32(52), 22);
    CHECK_EQ(u32(56), 0);
    CHECK_EQ(u64(64), (uint64_t)0xfffffff9u);
    CHECK_EQ(u64(72), (uint64_t)-7);
}

TEST(out_of_bounds_global_load_reports_thread_and_line) {
    Kernel k = wrap(R"(
	mul.wide.u32 %rd4, %r0, 4096;
	add.s64 %rd5, %rd1, %rd4;
	ld.global.u32 %r1, [%rd5];
)");
    CHECK_THROWS(run(k, 32, 32), MemoryFault, "out-of-bounds global load");
    CHECK_THROWS(run(k, 32, 32), MemoryFault, "thread 1");
}

TEST(out_of_bounds_shared_store_is_detected) {
    Kernel k = wrap(R"(
	mov.u32 %r2, 4096;
	st.shared.u32 [%r2], %r0;
	mov.u32 %r1, 0;
)");
    CHECK_THROWS(run(k, 32, 32), MemoryFault, "out-of-bounds shared store");
}

TEST(infinite_loop_hits_cycle_limit) {
    Kernel k = wrap(R"(
$L__spin:
	bra.uni $L__spin;
)");
    GpuConfig cfg;
    cfg.max_cycles = 10000;
    CHECK_THROWS(run(k, 32, 32, 0, cfg), SimError, "cycle limit");
}

TEST(launch_checks_argument_list) {
    Kernel k = wrap("\tmov.u32 %r1, 0;\n");
    Gpu gpu;
    CHECK_THROWS(gpu.launch(k, {1}, {32}, KernelArgs().ptr(0)), SimError, "expects 2 arguments");
    CHECK_THROWS(gpu.launch(k, {1}, {32}, KernelArgs().u32(0).u32(0)), SimError, "has size 4, expected 8");
}

TEST(multidimensional_thread_indices) {
    const char* src = R"(
.visible .entry ids(.param .u64 ids_out)
{
	.reg .b32 %r<12>;
	.reg .b64 %rd<6>;
	ld.param.u64 %rd1, [ids_out];
	mov.u32 %r1, %tid.x;
	mov.u32 %r2, %tid.y;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ctaid.y;
	mov.u32 %r5, %ntid.x;
	mov.u32 %r6, %nctaid.x;
	mad.lo.s32 %r7, %r4, %ntid.y, %r2;   // global y
	mad.lo.s32 %r8, %r3, %r5, %r1;       // global x
	mul.lo.s32 %r9, %r6, %r5;            // width
	mad.lo.s32 %r10, %r7, %r9, %r8;
	mul.wide.u32 %rd2, %r10, 4;
	add.s64 %rd3, %rd1, %rd2;
	shl.b32 %r11, %r7, 16;
	or.b32 %r11, %r11, %r8;
	st.global.u32 [%rd3], %r11;
	ret;
}
)";
    Kernel k = parse_ptx(src).kernel("ids");
    Gpu gpu;
    uint64_t out = gpu.alloc_zeroed<uint32_t>(24 * 12);
    gpu.launch(k, {3, 2}, {8, 6}, KernelArgs().ptr(out));
    auto v = gpu.download<uint32_t>(out, 24 * 12);
    for (uint32_t y = 0; y < 12; ++y)
        for (uint32_t x = 0; x < 24; ++x) CHECK_EQ(v[y * 24 + x], (y << 16) | x);
}

TEST(dependent_chain_pays_latency_independent_ops_do_not) {
    std::string chain, indep;
    for (int i = 0; i < 64; ++i) chain += "\tadd.s32 %r1, %r1, 1;\n";
    for (int i = 0; i < 64; ++i) indep += "\tadd.s32 %r" + std::to_string(2 + i % 20) + ", %r0, 1;\n";
    Kernel kc = wrap("\tmov.u32 %r1, 0;\n" + chain);
    Kernel ki = wrap("\tmov.u32 %r1, 0;\n" + indep);
    GpuConfig cfg;
    cfg.num_sms = 1;
    Out c = run(kc, 32, 32, 0, cfg);
    Out i = run(ki, 32, 32, 0, cfg);
    CHECK_EQ(c.v[0], 64u);
    CHECK(c.st.cycles >= (uint64_t)64 * cfg.lat_alu);
    CHECK(i.st.cycles < c.st.cycles / 2);
    CHECK(c.st.sched_cycles[kStallWait] > i.st.sched_cycles[kStallWait]);
}

TEST(more_warps_hide_alu_latency) {
    std::string chain;
    for (int i = 0; i < 64; ++i) chain += "\tadd.s32 %r1, %r1, 1;\n";
    Kernel k = wrap("\tmov.u32 %r1, 0;\n" + chain);
    GpuConfig cfg;
    cfg.num_sms = 1;
    cfg.schedulers_per_sm = 1;
    Out one = run(k, 32, 32, 0, cfg);
    Out four = run(k, 128, 128, 0, cfg);
    // Four independent warps interleave on one scheduler: ~4x the work in far less than 4x the time.
    CHECK(four.st.cycles < 2 * one.st.cycles);
    CHECK(four.st.ipc() > 2.5 * one.st.ipc());
}

TEST(dynamic_shared_memory_follows_static_shared) {
    const char* src = R"(
.extern .shared .align 16 .b8 dyn[];

.visible .entry rev(.param .u64 rev_out)
{
	.reg .b32 %r<10>;
	.reg .b64 %rd<5>;
	.shared .align 4 .b8 fixed[20];
	ld.param.u64 %rd1, [rev_out];
	mov.u32 %r1, %tid.x;
	mov.u32 %r2, dyn;
	shl.b32 %r3, %r1, 2;
	add.s32 %r4, %r2, %r3;
	mov.u32 %r9, -1;
	add.s32 %r5, %r1, %r9;
	st.shared.u32 [%r4], %r5;
	bar.sync 0;
	sub.s32 %r6, 63, %r1;
	shl.b32 %r7, %r6, 2;
	add.s32 %r8, %r2, %r7;
	ld.shared.u32 %r5, [%r8];
	mul.wide.u32 %rd2, %r1, 4;
	add.s64 %rd3, %rd1, %rd2;
	st.global.u32 [%rd3], %r5;
	ret;
}
)";
    Module m = parse_ptx(src);
    const Kernel& k = m.kernel("rev");
    CHECK(k.uses_dynamic_shared);
    CHECK_EQ(k.code[2].ops[1].imm, 32);  // 20 static bytes rounded up to 16-byte alignment
    CHECK_EQ(k.code[5].ops[1].imm, -1);  // ordinary immediates are untouched
    Gpu gpu;
    uint64_t out = gpu.alloc_zeroed<uint32_t>(64);
    LaunchOptions opt;
    opt.dynamic_shared_bytes = 256;
    gpu.launch(k, {1}, {64}, KernelArgs().ptr(out), opt);
    auto v = gpu.download<uint32_t>(out, 64);
    for (uint32_t t = 0; t < 64; ++t) CHECK_EQ(v[t], 62 - t);
    opt.dynamic_shared_bytes = 128;  // too small for 64 words: must fault
    CHECK_THROWS(gpu.launch(k, {1}, {64}, KernelArgs().ptr(out), opt), MemoryFault, "shared");
}
