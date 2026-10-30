#include "gpu.h"
#include "memory.h"
#include "test.h"

using namespace ptxsim;

static WarpAccess strided(uint64_t base, int stride_bytes, int size = 4, uint32_t mask = 0xffffffffu) {
    WarpAccess a;
    a.mask = mask;
    a.size = size;
    for (int l = 0; l < 32; ++l) a.addr[l] = base + (uint64_t)l * stride_bytes;
    return a;
}

TEST(coalescing_contiguous_floats_is_four_sectors) {
    auto r = coalesce(strided(0x1000, 4));
    CHECK_EQ(r.sectors.size(), (size_t)4);
    CHECK_EQ(r.lines, 1);
}

TEST(coalescing_misaligned_warp_touches_an_extra_sector) {
    auto r = coalesce(strided(0x1004, 4));
    CHECK_EQ(r.sectors.size(), (size_t)5);
    CHECK_EQ(r.lines, 2);
}

TEST(coalescing_degrades_with_stride) {
    CHECK_EQ(coalesce(strided(0x1000, 8)).sectors.size(), (size_t)8);
    CHECK_EQ(coalesce(strided(0x1000, 16)).sectors.size(), (size_t)16);
    CHECK_EQ(coalesce(strided(0x1000, 32)).sectors.size(), (size_t)32);
    auto r = coalesce(strided(0x1000, 128));
    CHECK_EQ(r.sectors.size(), (size_t)32);
    CHECK_EQ(r.lines, 32);
}

TEST(coalescing_ignores_inactive_lanes_and_merges_duplicates) {
    CHECK_EQ(coalesce(strided(0x1000, 128, 4, 0x1)).sectors.size(), (size_t)1);
    CHECK_EQ(coalesce(strided(0x1000, 0)).sectors.size(), (size_t)1);  // broadcast
    CHECK_EQ(coalesce(strided(0x1000, 16, 16)).sectors.size(), (size_t)16);  // float4 per lane
}

TEST(bank_conflicts_follow_word_stride) {
    CHECK_EQ(bank_conflict_degree(strided(0, 4)), 1);
    CHECK_EQ(bank_conflict_degree(strided(0, 8)), 2);
    CHECK_EQ(bank_conflict_degree(strided(0, 16)), 4);
    CHECK_EQ(bank_conflict_degree(strided(0, 128)), 32);  // column of a 32x32 float tile
    CHECK_EQ(bank_conflict_degree(strided(0, 132)), 1);   // column of a 32x33 padded tile
    CHECK_EQ(bank_conflict_degree(strided(0, 12)), 1);    // odd word stride is conflict-free
}

TEST(bank_conflicts_broadcast_same_word) {
    CHECK_EQ(bank_conflict_degree(strided(64, 0)), 1);
    WarpAccess a = strided(0, 0);
    for (int l = 16; l < 32; ++l) a.addr[l] = 128;  // two words, same bank
    CHECK_EQ(bank_conflict_degree(a), 2);
}

TEST(cache_hits_after_fill_and_tracks_sectors) {
    SectorCache c(4096, 2);  // 32 lines, 16 sets
    CHECK(!c.access(0x0, false).hit);
    CHECK(c.access(0x0, false).hit);
    CHECK(c.access(0x10, false).hit);   // same 32B sector
    CHECK(!c.access(0x20, false).hit);  // same line, other sector
    CHECK(c.access(0x20, false).hit);
}

TEST(cache_evicts_least_recently_used) {
    SectorCache c(4096, 2);
    const uint64_t set_stride = 128ull * c.sets();
    c.access(0 * set_stride, false);
    c.access(1 * set_stride, false);
    c.access(0 * set_stride, false);      // way 0 most recent
    c.access(2 * set_stride, false);      // evicts line 1
    CHECK(c.access(0 * set_stride, false).hit);
    CHECK(!c.access(1 * set_stride, false).hit);
}

TEST(cache_writes_back_dirty_sectors_on_eviction) {
    SectorCache c(4096, 1);
    const uint64_t set_stride = 128ull * c.sets();
    c.access(0x0, true);
    c.access(0x20, true);
    auto r = c.access(set_stride, false);
    CHECK_EQ(r.dirty_evicted_bytes, 64);
}

TEST(write_through_cache_invalidates_on_store) {
    SectorCache c(4096, 2);
    c.access(0x40, false);
    c.access(0x40, true, /*write_allocate=*/false);
    CHECK(!c.access(0x40, false).hit);
}

TEST(global_memory_bounds) {
    GlobalMemory m;
    uint64_t a = m.alloc(16);
    CHECK(m.valid(a, 16));
    CHECK(!m.valid(a, 1u << 20));
    CHECK(!m.valid(0, 4));
    uint32_t x = 7;
    CHECK_THROWS(m.write(0x10, &x, 4), MemoryFault, "out of bounds");
}

TEST(occupancy_limited_by_warps) {
    GpuConfig cfg;
    Occupancy o = compute_occupancy(cfg, 256, 16, 0);
    CHECK_EQ(o.warps_per_cta, 8);
    CHECK_EQ(o.ctas_per_sm, 6);
    CHECK_EQ(o.limiter, std::string("warps"));
    CHECK_NEAR(o.theoretical, 1.0, 1e-9);
}

TEST(occupancy_limited_by_registers) {
    GpuConfig cfg;
    // 128 regs/thread * 32 = 4096 per warp; 8 warps per CTA -> 32768 per CTA -> 2 CTAs
    Occupancy o = compute_occupancy(cfg, 256, 128, 0);
    CHECK_EQ(o.ctas_per_sm, 2);
    CHECK_EQ(o.limiter, std::string("registers"));
    CHECK_NEAR(o.theoretical, 16.0 / 48.0, 1e-9);
}

TEST(occupancy_limited_by_shared_memory_with_allocation_granularity) {
    GpuConfig cfg;
    Occupancy o = compute_occupancy(cfg, 64, 16, 20000);  // rounds to 20224 bytes -> 3 CTAs in 64 KiB
    CHECK_EQ(o.ctas_per_sm, 3);
    CHECK_EQ(o.limiter, std::string("shared memory"));
}

TEST(occupancy_limited_by_cta_slots) {
    GpuConfig cfg;
    Occupancy o = compute_occupancy(cfg, 32, 8, 0);
    CHECK_EQ(o.ctas_per_sm, 16);
    CHECK_EQ(o.limiter, std::string("ctas"));
}
