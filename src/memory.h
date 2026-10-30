// Memory system: functional device memory plus the timing-side models
// (sector coalescing, shared-memory banks, sectored set-associative caches).
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ptxsim {

struct MemoryFault : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Flat device memory with a bump allocator. Addresses start at kBase so that
// a null or small integer pointer always faults.
class GlobalMemory {
public:
    static constexpr uint64_t kBase = 0x10000000ull;

    uint64_t alloc(size_t bytes, size_t align = 256);
    void reset();
    bool valid(uint64_t addr, size_t size) const {
        return addr >= kBase && addr + size <= kBase + data_.size() && addr + size >= addr;
    }
    uint8_t* ptr(uint64_t addr) { return data_.data() + (addr - kBase); }
    void write(uint64_t addr, const void* src, size_t bytes);
    void read(uint64_t addr, void* dst, size_t bytes) const;
    size_t size() const { return data_.size(); }

private:
    std::vector<uint8_t> data_;
};

// Generic addresses in [kSharedWindow, kSharedWindow + 4GB) refer to the
// executing CTA's shared memory (cvta.shared / cvta.to.shared).
constexpr uint64_t kSharedWindow = 0x7f0000000000ull;

struct WarpAccess {
    uint32_t mask = 0;
    uint64_t addr[32];
    int size = 4;
};

// Unique 32-byte sectors touched by one warp-wide global access, in address order.
struct CoalesceResult {
    std::vector<uint64_t> sectors;   // sector index (addr / 32)
    int lines = 0;                   // distinct 128-byte lines
};
CoalesceResult coalesce(const WarpAccess& a, int sector_bytes = 32, int line_bytes = 128);

// Shared-memory bank conflict degree: the number of wavefronts needed to serve
// the access, i.e. the maximum number of distinct 32-bit words mapped to a
// single bank. Lanes reading the same word are served by broadcast.
int bank_conflict_degree(const WarpAccess& a, int banks = 32);

// Sectored, set-associative cache with LRU replacement. Tags only: data lives
// in GlobalMemory. Lines are `line_bytes` wide and split into 32-byte sectors.
class SectorCache {
public:
    SectorCache(size_t bytes, int assoc, int line_bytes = 128, int sector_bytes = 32);

    struct Result {
        bool hit;
        int dirty_evicted_bytes;  // bytes written back by the eviction this access caused
    };
    // Read: allocate on miss. Write: allocate and mark dirty (write-back cache)
    // when `write_allocate`, otherwise invalidate the sector (write-through L1).
    Result access(uint64_t addr, bool is_write, bool write_allocate = true);
    void flush();
    int sets() const { return sets_; }
    int assoc() const { return assoc_; }

private:
    struct Line {
        uint64_t tag = 0;
        bool valid = false;
        uint32_t sector_valid = 0;
        uint32_t sector_dirty = 0;
        uint64_t lru = 0;
    };
    int sets_, assoc_, line_bytes_, sector_bytes_;
    uint64_t tick_ = 0;
    std::vector<Line> lines_;
};

}  // namespace ptxsim
