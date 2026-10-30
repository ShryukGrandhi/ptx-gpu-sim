#include "memory.h"

#include <algorithm>
#include <sstream>

namespace ptxsim {

uint64_t GlobalMemory::alloc(size_t bytes, size_t align) {
    size_t off = (data_.size() + align - 1) / align * align;
    data_.resize(off + std::max<size_t>(bytes, 1), 0);
    return kBase + off;
}

void GlobalMemory::reset() { data_.clear(); }

void GlobalMemory::write(uint64_t addr, const void* src, size_t bytes) {
    if (!valid(addr, bytes)) {
        std::ostringstream os;
        os << "host write out of bounds at 0x" << std::hex << addr;
        throw MemoryFault(os.str());
    }
    std::memcpy(ptr(addr), src, bytes);
}

void GlobalMemory::read(uint64_t addr, void* dst, size_t bytes) const {
    if (!valid(addr, bytes)) {
        std::ostringstream os;
        os << "host read out of bounds at 0x" << std::hex << addr;
        throw MemoryFault(os.str());
    }
    std::memcpy(dst, data_.data() + (addr - kBase), bytes);
}

CoalesceResult coalesce(const WarpAccess& a, int sector_bytes, int line_bytes) {
    CoalesceResult r;
    r.sectors.reserve(64);
    for (int lane = 0; lane < 32; ++lane) {
        if (!(a.mask >> lane & 1)) continue;
        uint64_t first = a.addr[lane] / sector_bytes;
        uint64_t last = (a.addr[lane] + a.size - 1) / sector_bytes;
        for (uint64_t s = first; s <= last; ++s) r.sectors.push_back(s);
    }
    std::sort(r.sectors.begin(), r.sectors.end());
    r.sectors.erase(std::unique(r.sectors.begin(), r.sectors.end()), r.sectors.end());
    const uint64_t per_line = line_bytes / sector_bytes;
    uint64_t prev = ~0ull;
    for (uint64_t s : r.sectors) {
        if (s / per_line != prev) { ++r.lines; prev = s / per_line; }
    }
    return r;
}

int bank_conflict_degree(const WarpAccess& a, int banks) {
    // Collect distinct word addresses, then count how many land in each bank.
    uint64_t words[32 * 4];
    int n = 0;
    const int nwords = std::max(1, std::min(4, a.size / 4));
    for (int lane = 0; lane < 32; ++lane) {
        if (!(a.mask >> lane & 1)) continue;
        for (int w = 0; w < nwords; ++w) words[n++] = a.addr[lane] / 4 + w;
    }
    std::sort(words, words + n);
    n = (int)(std::unique(words, words + n) - words);
    int per_bank[64] = {0};
    int degree = 0;
    for (int i = 0; i < n; ++i) degree = std::max(degree, ++per_bank[words[i] % banks]);
    return degree;
}

SectorCache::SectorCache(size_t bytes, int assoc, int line_bytes, int sector_bytes)
    : assoc_(assoc), line_bytes_(line_bytes), sector_bytes_(sector_bytes) {
    sets_ = std::max<int>(1, (int)(bytes / line_bytes / assoc));
    lines_.resize((size_t)sets_ * assoc_);
}

void SectorCache::flush() {
    for (auto& l : lines_) l = Line{};
    tick_ = 0;
}

SectorCache::Result SectorCache::access(uint64_t addr, bool is_write, bool write_allocate) {
    const uint64_t line_addr = addr / line_bytes_;
    const int set = (int)(line_addr % sets_);
    const uint32_t sector_bit = 1u << ((addr % line_bytes_) / sector_bytes_);
    Line* base = &lines_[(size_t)set * assoc_];
    ++tick_;
    for (int w = 0; w < assoc_; ++w) {
        Line& l = base[w];
        if (l.valid && l.tag == line_addr) {
            bool hit = (l.sector_valid & sector_bit) != 0;
            if (is_write && !write_allocate) {
                l.sector_valid &= ~sector_bit;
                return {false, 0};
            }
            l.lru = tick_;
            l.sector_valid |= sector_bit;
            if (is_write) l.sector_dirty |= sector_bit;
            return {hit, 0};
        }
    }
    if (is_write && !write_allocate) return {false, 0};
    Line* victim = base;
    for (int w = 0; w < assoc_; ++w) {
        if (!base[w].valid) { victim = &base[w]; break; }
        if (base[w].lru < victim->lru) victim = &base[w];
    }
    int written_back = victim->valid ? __builtin_popcount(victim->sector_dirty) * sector_bytes_ : 0;
    victim->valid = true;
    victim->tag = line_addr;
    victim->sector_valid = sector_bit;
    victim->sector_dirty = is_write ? sector_bit : 0;
    victim->lru = tick_;
    return {false, written_back};
}

}  // namespace ptxsim
