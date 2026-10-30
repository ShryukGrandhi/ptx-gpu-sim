// SIMT execution engine.
//
// Each SM holds resident CTAs and their warps. Every cycle, each warp
// scheduler picks one ready warp (greedy-then-oldest or loose round-robin)
// and issues its next instruction. Instructions execute functionally at
// issue; the timing model decides when their results become visible to
// dependent instructions (per-register scoreboard), how long the load/store
// unit stays busy, and how much L1/L2/DRAM traffic they generate.
//
// Branch divergence uses a per-warp reconvergence stack. Divergent paths
// merge at the immediate post-dominator computed in cfg.cpp.
#include "gpu.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace ptxsim {

const char* stall_reason_name(int r) {
    switch (r) {
        case kIssued: return "issued";
        case kStallLongScoreboard: return "long_scoreboard";
        case kStallShortScoreboard: return "short_scoreboard";
        case kStallWait: return "wait";
        case kStallBarrier: return "barrier";
        case kStallMemThrottle: return "mem_throttle";
        case kStallIdle: return "idle";
        default: return "?";
    }
}

double KernelStats::stall_fraction(int r) const {
    uint64_t total = 0;
    for (auto c : sched_cycles) total += c;
    return total ? (double)sched_cycles[r] / total : 0.0;
}

std::string KernelStats::report() const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "kernel " << kernel << "\n"
       << "  cycles                 " << cycles << "\n"
       << "  warp instructions      " << warp_instructions << "  (IPC " << ipc() << ")\n"
       << "  SIMD efficiency        " << simd_efficiency() << "\n"
       << "  branches (divergent)   " << branches << " (" << divergent_branches << ")\n"
       << "  global ld sectors/req  " << gld_sectors_per_request() << "  (" << gld_requests << " requests)\n"
       << "  global st sectors/req  " << gst_sectors_per_request() << "  (" << gst_requests << " requests)\n"
       << "  L1 hit rate            " << l1_hit_rate() << "\n"
       << "  L2 hit rate            " << l2_hit_rate() << "\n"
       << "  DRAM read/write bytes  " << dram_read_bytes << " / " << dram_write_bytes << "\n"
       << "  shared requests        " << shared_requests << "  (bank conflict wavefronts " << bank_conflicts()
       << ")\n"
       << "  atomics (serialized)   " << atomic_requests << " (" << atomic_serialized << ")\n"
       << "  occupancy              theoretical " << occupancy.theoretical << " (" << occupancy.ctas_per_sm
       << " CTAs/SM, limited by " << occupancy.limiter << "), achieved " << achieved_occupancy << "\n"
       << "  scheduler state       ";
    for (int r = 0; r < kNumStallReasons; ++r) os << " " << stall_reason_name(r) << "=" << stall_fraction(r);
    os << "\n";
    return os.str();
}

Occupancy compute_occupancy(const GpuConfig& cfg, int threads_per_cta, int regs_per_thread, int smem_per_cta) {
    Occupancy o;
    o.warps_per_cta = (threads_per_cta + 31) / 32;
    o.regs_per_thread = regs_per_thread;
    o.smem_per_cta = smem_per_cta;
    auto round_up = [](int v, int unit) { return (v + unit - 1) / unit * unit; };

    int by_warps = cfg.max_warps_per_sm / o.warps_per_cta;
    int by_ctas = cfg.max_ctas_per_sm;
    int regs_per_warp = round_up(regs_per_thread * 32, cfg.reg_alloc_unit);
    int by_regs = regs_per_warp > 0 ? cfg.regs_per_sm / (regs_per_warp * o.warps_per_cta) : by_ctas;
    int smem = round_up(smem_per_cta, cfg.smem_alloc_unit);
    int by_smem = smem > 0 ? cfg.smem_per_sm / smem : by_ctas;

    o.ctas_per_sm = by_warps;
    o.limiter = "warps";
    if (by_ctas < o.ctas_per_sm) { o.ctas_per_sm = by_ctas; o.limiter = "ctas"; }
    if (by_regs < o.ctas_per_sm) { o.ctas_per_sm = by_regs; o.limiter = "registers"; }
    if (by_smem < o.ctas_per_sm) { o.ctas_per_sm = by_smem; o.limiter = "shared memory"; }
    o.ctas_per_sm = std::max(o.ctas_per_sm, 0);
    o.active_warps_per_sm = o.ctas_per_sm * o.warps_per_cta;
    o.theoretical = (double)o.active_warps_per_sm / cfg.max_warps_per_sm;
    return o;
}

namespace {

constexpr uint64_t kNever = std::numeric_limits<uint64_t>::max();

inline float as_f32(uint64_t b) {
    uint32_t u = (uint32_t)b;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline uint64_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline uint64_t trunc_to(uint64_t v, int bytes) { return bytes >= 8 ? v : v & ((1ull << (8 * bytes)) - 1); }
inline int64_t sext(uint64_t v, int bytes) {
    if (bytes >= 8) return (int64_t)v;
    int shift = 64 - 8 * bytes;
    return (int64_t)(v << shift) >> shift;
}

struct SimtEntry {
    int pc;
    int rpc;  // reconvergence pc; -1 for the base entry
    uint32_t mask;
};

struct Cta {
    bool active = false;
    Dim3 id;
    std::vector<uint8_t> shared;
    std::vector<int> warps;
    int live_warps = 0;
    int arrived = 0;
};

struct Warp {
    bool active = false;
    int cta = -1;
    int warp_in_cta = 0;
    uint32_t valid = 0;
    uint32_t exited = 0;
    std::vector<SimtEntry> stack;
    std::vector<uint64_t> regs;      // [reg * 32 + lane]
    std::vector<uint64_t> ready;     // cycle at which each register's pending write lands
    std::vector<uint8_t> ready_kind; // stall reason charged while waiting on it
    bool at_barrier = false;
    uint64_t age = 0;
};

struct Sm {
    std::vector<Warp> warps;
    std::vector<Cta> ctas;
    int resident_ctas = 0;
    int resident_warps = 0;
    SectorCache l1;
    uint64_t lsu_free = 0;
    uint64_t wake = kNever;
    std::vector<int> greedy;    // per scheduler, GTO current warp
    std::vector<int> rr_last;   // per scheduler, LRR last issued
    std::vector<int> state;     // per scheduler, StallReason for the last evaluated cycle
    uint64_t busy_cycles = 0;
    uint64_t warp_cycles = 0;

    explicit Sm(const GpuConfig& c) : l1(c.l1_bytes, c.l1_assoc) {}
};

class Engine {
public:
    Engine(const GpuConfig& cfg, GlobalMemory& mem, const Kernel& k, Dim3 grid, Dim3 block,
           std::vector<uint8_t> params, const Occupancy& occ, int smem_per_cta, KernelStats& st)
        : cfg_(cfg), mem_(mem), k_(k), grid_(grid), block_(block), params_(std::move(params)), occ_(occ),
          smem_per_cta_(smem_per_cta), st_(st), l2_(cfg.l2_bytes, cfg.l2_assoc) {
        nregs_ = (int)k.regs.size();
        for (int i = 0; i < cfg.num_sms; ++i) {
            sms_.emplace_back(cfg);
            Sm& sm = sms_.back();
            sm.warps.resize(cfg.max_warps_per_sm);
            sm.ctas.resize(cfg.max_ctas_per_sm);
            sm.greedy.assign(cfg.schedulers_per_sm, -1);
            sm.rr_last.assign(cfg.schedulers_per_sm, -1);
            sm.state.assign(cfg.schedulers_per_sm, kStallIdle);
        }
        total_ctas_ = grid.count();
        threads_per_cta_ = block.count();
    }

    void run() {
        uint64_t now = 0;
        while (true) {
            dispatch(now);
            if (next_cta_ == total_ctas_ && done_ctas_ == total_ctas_) break;
            for (auto& sm : sms_)
                if (sm.wake <= now) step(sm, now);
            uint64_t next = kNever;
            for (auto& sm : sms_)
                if (sm.resident_warps > 0) next = std::min(next, sm.wake);
            if (done_ctas_ == total_ctas_ && next_cta_ == total_ctas_) { now += 1; account(1); break; }
            if (next == kNever) {
                if (next_cta_ < total_ctas_) next = now + 1;  // a CTA slot just freed up
                else throw SimError("deadlock in kernel " + k_.name +
                                    ": all resident warps blocked (barrier that can never complete?)");
            }
            next = std::max(next, now + 1);
            account(next - now);
            now = next;
            if (now > cfg_.max_cycles) throw SimError("cycle limit exceeded in kernel " + k_.name);
        }
        st_.cycles = now;
        uint64_t busy = 0, warp_cycles = 0;
        for (auto& sm : sms_) { busy += sm.busy_cycles; warp_cycles += sm.warp_cycles; }
        st_.achieved_occupancy = busy ? (double)warp_cycles / busy / cfg_.max_warps_per_sm : 0.0;
    }

private:
    const GpuConfig& cfg_;
    GlobalMemory& mem_;
    const Kernel& k_;
    Dim3 grid_, block_;
    std::vector<uint8_t> params_;
    Occupancy occ_;
    int smem_per_cta_;
    KernelStats& st_;
    SectorCache l2_;
    double l2_free_ = 0, dram_free_ = 0;  // bandwidth queues (cycle at which the next transfer can start)
    std::vector<Sm> sms_;
    int nregs_ = 0;
    uint32_t total_ctas_ = 0, next_cta_ = 0, done_ctas_ = 0;
    uint32_t threads_per_cta_ = 0;
    uint64_t age_counter_ = 0;
    int dispatch_rr_ = 0;

    void account(uint64_t dt) {
        for (auto& sm : sms_) {
            if (sm.resident_warps > 0) {
                sm.busy_cycles += dt;
                sm.warp_cycles += (uint64_t)sm.resident_warps * dt;
            }
            for (int s : sm.state) {
                if (s != kIssued) { st_.sched_cycles[s] += dt; continue; }
                st_.sched_cycles[kIssued] += 1;
                if (dt > 1) st_.sched_cycles[sm.resident_warps ? kStallWait : kStallIdle] += dt - 1;
            }
        }
    }

    // ---------------------------------------------------------------- dispatch
    void dispatch(uint64_t now) {
        while (next_cta_ < total_ctas_) {
            bool placed = false;
            for (int tries = 0; tries < cfg_.num_sms && next_cta_ < total_ctas_; ++tries) {
                Sm& sm = sms_[dispatch_rr_];
                dispatch_rr_ = (dispatch_rr_ + 1) % cfg_.num_sms;
                if (sm.resident_ctas < occ_.ctas_per_sm) {
                    launch_cta(sm, next_cta_++);
                    sm.wake = std::min(sm.wake, now);
                    placed = true;
                }
            }
            if (!placed) break;
        }
    }

    void launch_cta(Sm& sm, uint32_t linear_id) {
        int slot = 0;
        while (sm.ctas[slot].active) ++slot;
        Cta& c = sm.ctas[slot];
        c.active = true;
        c.id.x = linear_id % grid_.x;
        c.id.y = (linear_id / grid_.x) % grid_.y;
        c.id.z = linear_id / (grid_.x * grid_.y);
        c.shared.assign(smem_per_cta_, 0);
        c.warps.clear();
        c.arrived = 0;
        const int nw = occ_.warps_per_cta;
        c.live_warps = nw;
        int w = 0;
        for (int i = 0; i < nw; ++i) {
            while (sm.warps[w].active) ++w;
            Warp& wp = sm.warps[w];
            wp.active = true;
            wp.cta = slot;
            wp.warp_in_cta = i;
            uint32_t remaining = threads_per_cta_ - i * 32;
            wp.valid = remaining >= 32 ? 0xffffffffu : ((1u << remaining) - 1);
            wp.exited = 0;
            wp.stack.assign(1, SimtEntry{0, -1, wp.valid});
            wp.regs.assign((size_t)nregs_ * 32, 0);
            wp.ready.assign(nregs_, 0);
            wp.ready_kind.assign(nregs_, kStallWait);
            wp.at_barrier = false;
            wp.age = age_counter_++;
            c.warps.push_back(w);
        }
        sm.resident_ctas++;
        sm.resident_warps += nw;
    }

    // ---------------------------------------------------------------- scheduling
    bool needs_lsu(const Instr& in) const {
        return in.unit == Unit::MemGlobal || in.unit == Unit::MemShared || in.unit == Unit::MemGeneric ||
               in.unit == Unit::Atomic;
    }

    // Returns kIssued if the warp can issue now, otherwise the stall reason and
    // the earliest cycle at which it might become ready.
    int check(const Sm& sm, const Warp& w, uint64_t now, uint64_t& hint) const {
        if (w.at_barrier) { hint = kNever; return kStallBarrier; }
        const Instr& in = k_.code[w.stack.back().pc];
        uint64_t t = 0;
        int kind = kStallWait;
        for (int r : in.uses) if (w.ready[r] > t) { t = w.ready[r]; kind = w.ready_kind[r]; }
        for (int r : in.defs) if (w.ready[r] > t) { t = w.ready[r]; kind = w.ready_kind[r]; }
        if (t > now) { hint = t; return kind; }
        if (needs_lsu(in) && sm.lsu_free > now) { hint = sm.lsu_free; return kStallMemThrottle; }
        return kIssued;
    }

    void step(Sm& sm, uint64_t now) {
        const int S = cfg_.schedulers_per_sm;
        const int W = cfg_.max_warps_per_sm;
        uint64_t wake = kNever;
        bool issued_any = false;
        for (int s = 0; s < S; ++s) {
            int pick = -1;
            int oldest = -1;
            int oldest_reason = kStallIdle;
            uint64_t hint;
            if (cfg_.policy == SchedPolicy::GreedyThenOldest) {
                // Keep issuing from the same warp while it can; otherwise take the oldest ready warp.
                const int g = sm.greedy[s];
                bool greedy_ready = false;
                for (int w = s; w < W; w += S) {
                    const Warp& wp = sm.warps[w];
                    if (!wp.active) continue;
                    int r = check(sm, wp, now, hint);
                    if (r != kIssued) wake = std::min(wake, hint);
                    if (oldest < 0 || wp.age < sm.warps[oldest].age) { oldest = w; oldest_reason = r; }
                    if (r == kIssued) {
                        if (w == g) greedy_ready = true;
                        if (pick < 0 || wp.age < sm.warps[pick].age) pick = w;
                    }
                }
                if (greedy_ready) pick = g;
                if (pick >= 0) sm.greedy[s] = pick;
            } else {
                int slots = (W - s + S - 1) / S;
                int start = sm.rr_last[s] < 0 ? 0 : ((sm.rr_last[s] - s) / S + 1) % slots;
                for (int i = 0; i < slots; ++i) {
                    int w = s + ((start + i) % slots) * S;
                    const Warp& wp = sm.warps[w];
                    if (!wp.active) continue;
                    int r = check(sm, wp, now, hint);
                    if (r != kIssued) wake = std::min(wake, hint);
                    if (oldest < 0 || wp.age < sm.warps[oldest].age) { oldest = w; oldest_reason = r; }
                    if (pick < 0 && r == kIssued) pick = w;
                }
                if (pick >= 0) sm.rr_last[s] = pick;
            }
            if (pick >= 0) {
                sm.state[s] = kIssued;
                issued_any = true;
                issue(sm, pick, now);
            } else {
                sm.state[s] = oldest < 0 ? kStallIdle : oldest_reason;
            }
        }
        sm.wake = issued_any ? now + 1 : wake;
    }

    // ---------------------------------------------------------------- operands
    uint64_t special(const Sm& sm, const Warp& w, Special s, int lane) const {
        uint32_t lin = w.warp_in_cta * 32 + lane;
        const Dim3& id = sm.ctas[w.cta].id;
        switch (s) {
            case Special::TidX: return lin % block_.x;
            case Special::TidY: return (lin / block_.x) % block_.y;
            case Special::TidZ: return lin / (block_.x * block_.y);
            case Special::NtidX: return block_.x;
            case Special::NtidY: return block_.y;
            case Special::NtidZ: return block_.z;
            case Special::CtaidX: return id.x;
            case Special::CtaidY: return id.y;
            case Special::CtaidZ: return id.z;
            case Special::NctaidX: return grid_.x;
            case Special::NctaidY: return grid_.y;
            case Special::NctaidZ: return grid_.z;
            case Special::LaneId: return (uint64_t)lane;
            case Special::WarpId: return (uint64_t)w.warp_in_cta;
        }
        return 0;
    }

    uint64_t read(const Sm& sm, const Warp& w, const Operand& o, int lane) const {
        switch (o.kind) {
            case Operand::Kind::Reg: return w.regs[(size_t)o.reg * 32 + lane];
            case Operand::Kind::Imm: return (uint64_t)o.imm;
            case Operand::Kind::Special: return special(sm, w, o.special, lane);
            default: return 0;
        }
    }

    uint64_t address(const Warp& w, const Operand& o, int lane) const {
        uint64_t base = o.reg >= 0 ? w.regs[(size_t)o.reg * 32 + lane] : 0;
        return base + (uint64_t)o.imm;
    }

    [[noreturn]] void fault(const Sm& sm, const Warp& w, int lane, int pc, const std::string& what,
                            uint64_t addr) const {
        const Dim3& id = sm.ctas[w.cta].id;
        std::ostringstream os;
        os << what << " at address 0x" << std::hex << addr << std::dec << " in kernel " << k_.name << ", CTA ("
           << id.x << "," << id.y << "," << id.z << "), thread " << (w.warp_in_cta * 32 + lane) << ", pc " << pc
           << " (line " << k_.code[pc].line << ": " << k_.code[pc].text << ")";
        throw MemoryFault(os.str());
    }

    // ---------------------------------------------------------------- ALU
    uint64_t alu(const Instr& in, uint64_t a, uint64_t b, uint64_t c) const {
        const Type t = in.type;
        const int nb = type_bytes(t);
        const bool sg = type_is_signed(t);
        const bool fl = type_is_float(t);
        switch (in.op) {
            case Op::Mov: return a;
            case Op::Add: return fl ? f32_bits(as_f32(a) + as_f32(b)) : trunc_to(a + b, nb);
            case Op::Sub: return fl ? f32_bits(as_f32(a) - as_f32(b)) : trunc_to(a - b, nb);
            case Op::Mul:
            case Op::Mad: {
                uint64_t p;
                if (fl) {
                    if (in.op == Op::Mad) return f32_bits(std::fma(as_f32(a), as_f32(b), as_f32(c)));
                    return f32_bits(as_f32(a) * as_f32(b));
                }
                if (in.mul == MulMode::Wide) {
                    p = sg ? (uint64_t)(sext(a, nb) * sext(b, nb)) : trunc_to(a, nb) * trunc_to(b, nb);
                    p = trunc_to(p, nb * 2);
                    return in.op == Op::Mad ? trunc_to(p + c, nb * 2) : p;
                }
                if (in.mul == MulMode::Hi) {
                    if (sg) {
                        __int128 q = (__int128)sext(a, nb) * sext(b, nb);
                        p = (uint64_t)(q >> (8 * nb));
                    } else {
                        unsigned __int128 q = (unsigned __int128)trunc_to(a, nb) * trunc_to(b, nb);
                        p = (uint64_t)(q >> (8 * nb));
                    }
                } else {
                    p = a * b;
                }
                return in.op == Op::Mad ? trunc_to(p + c, nb) : trunc_to(p, nb);
            }
            case Op::Fma: return f32_bits(std::fma(as_f32(a), as_f32(b), as_f32(c)));
            case Op::Div:
            case Op::Rem: {
                if (fl) return f32_bits(in.op == Op::Div ? as_f32(a) / as_f32(b) : std::fmod(as_f32(a), as_f32(b)));
                if (trunc_to(b, nb) == 0) return trunc_to(~0ull, nb);
                if (sg) {
                    int64_t x = sext(a, nb), y = sext(b, nb);
                    return trunc_to((uint64_t)(in.op == Op::Div ? x / y : x % y), nb);
                }
                uint64_t x = trunc_to(a, nb), y = trunc_to(b, nb);
                return in.op == Op::Div ? x / y : x % y;
            }
            case Op::Min:
            case Op::Max: {
                bool want_min = in.op == Op::Min;
                if (fl) return f32_bits(want_min ? std::fmin(as_f32(a), as_f32(b)) : std::fmax(as_f32(a), as_f32(b)));
                if (sg) {
                    int64_t x = sext(a, nb), y = sext(b, nb);
                    return trunc_to((uint64_t)(want_min ? std::min(x, y) : std::max(x, y)), nb);
                }
                uint64_t x = trunc_to(a, nb), y = trunc_to(b, nb);
                return want_min ? std::min(x, y) : std::max(x, y);
            }
            case Op::Neg: return fl ? (a ^ 0x80000000ull) : trunc_to(0 - a, nb);
            case Op::Abs: return fl ? (a & 0x7fffffffull) : trunc_to((uint64_t)std::llabs(sext(a, nb)), nb);
            case Op::And: return t == Type::Pred ? (a & b & 1) : trunc_to(a & b, nb);
            case Op::Or: return t == Type::Pred ? ((a | b) & 1) : trunc_to(a | b, nb);
            case Op::Xor: return t == Type::Pred ? ((a ^ b) & 1) : trunc_to(a ^ b, nb);
            case Op::Not: return t == Type::Pred ? (~a & 1) : trunc_to(~a, nb);
            case Op::Shl: {
                uint64_t s = trunc_to(b, 4);
                return s >= (uint64_t)(8 * nb) ? 0 : trunc_to(a << s, nb);
            }
            case Op::Shr: {
                uint64_t s = trunc_to(b, 4);
                if (sg) return trunc_to((uint64_t)(sext(a, nb) >> std::min<uint64_t>(s, 8 * nb - 1)), nb);
                return s >= (uint64_t)(8 * nb) ? 0 : trunc_to(a, nb) >> s;
            }
            case Op::Setp: {
                int r;
                if (fl) {
                    float x = as_f32(a), y = as_f32(b);
                    r = x < y ? -1 : x > y ? 1 : 0;
                    if (std::isnan(x) || std::isnan(y)) return in.cmp == Cmp::Ne ? 1 : 0;
                } else if (sg && !in.cmp_unsigned) {
                    int64_t x = sext(a, nb), y = sext(b, nb);
                    r = x < y ? -1 : x > y ? 1 : 0;
                } else {
                    uint64_t x = trunc_to(a, nb), y = trunc_to(b, nb);
                    r = x < y ? -1 : x > y ? 1 : 0;
                }
                switch (in.cmp) {
                    case Cmp::Eq: return r == 0;
                    case Cmp::Ne: return r != 0;
                    case Cmp::Lt: return r < 0;
                    case Cmp::Le: return r <= 0;
                    case Cmp::Gt: return r > 0;
                    case Cmp::Ge: return r >= 0;
                }
                return 0;
            }
            case Op::Selp: return (c & 1) ? a : b;
            case Op::Sqrt: return f32_bits(std::sqrt(as_f32(a)));
            case Op::Rsqrt: return f32_bits(1.0f / std::sqrt(as_f32(a)));
            case Op::Rcp: return f32_bits(1.0f / as_f32(a));
            case Op::Ex2: return f32_bits(std::exp2(as_f32(a)));
            case Op::Lg2: return f32_bits(std::log2(as_f32(a)));
            case Op::Cvt: {
                const Type dt = in.type, srct = in.type2;
                if (type_is_float(srct) && type_is_float(dt)) return a;
                if (type_is_float(srct)) {
                    float x = std::trunc(as_f32(a));
                    if (std::isnan(x)) return 0;
                    if (type_is_signed(dt)) {
                        double lim = std::ldexp(1.0, 8 * type_bytes(dt) - 1);
                        double v = std::max(-lim, std::min((double)x, lim - 1));
                        return trunc_to((uint64_t)(int64_t)v, type_bytes(dt));
                    }
                    double lim = std::ldexp(1.0, 8 * type_bytes(dt));
                    double v = std::max(0.0, std::min((double)x, lim - 1));
                    return trunc_to((uint64_t)v, type_bytes(dt));
                }
                if (type_is_float(dt))
                    return type_is_signed(srct) ? f32_bits((float)sext(a, type_bytes(srct)))
                                                : f32_bits((float)trunc_to(a, type_bytes(srct)));
                uint64_t v = type_is_signed(srct) ? (uint64_t)sext(a, type_bytes(srct)) : trunc_to(a, type_bytes(srct));
                return trunc_to(v, type_bytes(dt));
            }
            case Op::Cvta:
                if (in.space == Space::Shared) return in.cvta_to ? a - kSharedWindow : a + kSharedWindow;
                return a;
            default: return 0;
        }
    }

    // ---------------------------------------------------------------- memory helpers
    uint64_t load_bytes(Sm& sm, Warp& w, Space sp, uint64_t addr, int size, int lane, int pc) {
        uint64_t v = 0;
        if (sp == Space::Shared) {
            auto& sh = sm.ctas[w.cta].shared;
            if (addr + size > sh.size() || addr + size < addr) fault(sm, w, lane, pc, "out-of-bounds shared load", addr);
            std::memcpy(&v, sh.data() + addr, size);
        } else if (sp == Space::Param) {
            if (addr + size > params_.size()) fault(sm, w, lane, pc, "out-of-bounds param load", addr);
            std::memcpy(&v, params_.data() + addr, size);
        } else {
            if (!mem_.valid(addr, size)) fault(sm, w, lane, pc, "out-of-bounds global load", addr);
            std::memcpy(&v, mem_.ptr(addr), size);
        }
        return v;
    }

    void store_bytes(Sm& sm, Warp& w, Space sp, uint64_t addr, uint64_t v, int size, int lane, int pc) {
        if (sp == Space::Shared) {
            auto& sh = sm.ctas[w.cta].shared;
            if (addr + size > sh.size() || addr + size < addr) fault(sm, w, lane, pc, "out-of-bounds shared store", addr);
            std::memcpy(sh.data() + addr, &v, size);
        } else if (sp == Space::Param) {
            fault(sm, w, lane, pc, "store to param space", addr);
        } else {
            if (!mem_.valid(addr, size)) fault(sm, w, lane, pc, "out-of-bounds global store", addr);
            std::memcpy(mem_.ptr(addr), &v, size);
        }
    }

    // Resolve generic addresses to a concrete space (per lane).
    Space resolve(Space sp, uint64_t& addr) const {
        if (sp != Space::Generic) return sp;
        if (addr >= kSharedWindow && addr < kSharedWindow + (1ull << 32)) {
            addr -= kSharedWindow;
            return Space::Shared;
        }
        return Space::Global;
    }

    // Timing for a warp-wide global access. Returns the cycle at which the
    // data is available (loads) or the request is accepted (stores).
    uint64_t global_timing(Sm& sm, const WarpAccess& acc, bool is_store, uint64_t now) {
        CoalesceResult cr = coalesce(acc);
        sm.lsu_free = now + std::max(1, cr.lines);
        uint64_t done = now + cfg_.lat_l1;
        if (is_store) { st_.gst_requests++; st_.gst_sectors += cr.sectors.size(); }
        else { st_.gld_requests++; st_.gld_sectors += cr.sectors.size(); }
        for (uint64_t s : cr.sectors) {
            uint64_t addr = s * 32;
            if (!is_store) {
                auto r1 = sm.l1.access(addr, false);
                if (r1.hit) { st_.l1_hits++; continue; }
                st_.l1_misses++;
            } else {
                sm.l1.access(addr, true, /*write_allocate=*/false);
            }
            double start = std::max((double)now, l2_free_);
            l2_free_ = start + 32.0 / cfg_.l2_bytes_per_cycle;
            auto r2 = l2_.access(addr, is_store);
            if (r2.dirty_evicted_bytes) {
                st_.dram_write_bytes += r2.dirty_evicted_bytes;
                dram_free_ = std::max(dram_free_, start) + r2.dirty_evicted_bytes / cfg_.dram_bytes_per_cycle;
            }
            if (is_store) continue;
            uint64_t ready;
            if (r2.hit) {
                st_.l2_hits++;
                ready = (uint64_t)start + cfg_.lat_l2;
            } else {
                st_.l2_misses++;
                st_.dram_read_bytes += 32;
                double dstart = std::max(start, dram_free_);
                dram_free_ = dstart + 32.0 / cfg_.dram_bytes_per_cycle;
                ready = (uint64_t)dstart + cfg_.lat_dram;
            }
            done = std::max(done, ready);
        }
        return done;
    }

    uint64_t shared_timing(Sm& sm, const WarpAccess& acc, uint64_t now) {
        int degree = std::max(1, bank_conflict_degree(acc, cfg_.smem_banks));
        st_.shared_requests++;
        st_.shared_wavefronts += degree;
        sm.lsu_free = now + degree;
        return now + degree - 1 + cfg_.lat_shared;
    }

    void set_ready(Warp& w, const Instr& in, uint64_t t, int kind) {
        for (int r : in.defs) { w.ready[r] = t; w.ready_kind[r] = (uint8_t)kind; }
    }

    // ---------------------------------------------------------------- issue
    void issue(Sm& sm, int wi, uint64_t now) {
        Warp& w = sm.warps[wi];
        SimtEntry& top = w.stack.back();
        const int pc = top.pc;
        const Instr& in = k_.code[pc];
        const uint32_t active = top.mask & ~w.exited;
        uint32_t exec = active;
        if (in.guard >= 0) {
            uint32_t pm = 0;
            const uint64_t* p = &w.regs[(size_t)in.guard * 32];
            for (int l = 0; l < 32; ++l) if (p[l] & 1) pm |= 1u << l;
            exec &= in.guard_neg ? ~pm : pm;
        }
        st_.warp_instructions++;
        st_.thread_instructions += __builtin_popcount(active);

        switch (in.op) {
            case Op::Bra: {
                st_.branches++;
                const int target = in.target;
                if (exec == active) top.pc = target;
                else if (exec == 0) top.pc = pc + 1;
                else {
                    st_.divergent_branches++;
                    const int rpc = k_.reconverge_pc[pc];
                    top.pc = rpc;
                    w.stack.push_back({pc + 1, rpc, active & ~exec});
                    w.stack.push_back({target, rpc, exec});
                }
                break;
            }
            case Op::Ret:
            case Op::Exit:
                w.exited |= exec;
                top.pc = pc + 1;
                break;
            case Op::Bar: {
                top.pc = pc + 1;
                w.at_barrier = true;
                Cta& c = sm.ctas[w.cta];
                c.arrived++;
                maybe_release_barrier(sm, c);
                break;
            }
            case Op::Nop:
                top.pc = pc + 1;
                break;
            case Op::Ld: case Op::St:
                exec_ldst(sm, w, in, pc, exec, now);
                w.stack.back().pc = pc + 1;
                break;
            case Op::Atom: case Op::Red:
                exec_atomic(sm, w, in, pc, exec, now);
                w.stack.back().pc = pc + 1;
                break;
            case Op::Shfl:
                exec_shfl(sm, w, in, exec, now);
                w.stack.back().pc = pc + 1;
                break;
            default:
                exec_alu(sm, w, in, exec, now);
                w.stack.back().pc = pc + 1;
                break;
        }
        reconverge(sm, wi);
    }

    void exec_alu(Sm& sm, Warp& w, const Instr& in, uint32_t exec, uint64_t now) {
        const Operand& d = in.ops[0];
        uint64_t* dst = d.kind == Operand::Kind::Reg ? &w.regs[(size_t)d.reg * 32] : nullptr;
        uint64_t* dst2 = in.dst2 >= 0 ? &w.regs[(size_t)in.dst2 * 32] : nullptr;
        // Operand order differs: selp d, a, b, c ; setp d, a, b ; mad/fma d, a, b, c.
        for (int l = 0; l < 32; ++l) {
            if (!(exec >> l & 1)) continue;
            uint64_t a = in.nops > 1 ? read(sm, w, in.ops[1], l) : 0;
            uint64_t b = in.nops > 2 ? read(sm, w, in.ops[2], l) : 0;
            uint64_t c = in.nops > 3 ? read(sm, w, in.ops[3], l) : 0;
            uint64_t r = alu(in, a, b, c);
            if (dst) dst[l] = r;
            if (dst2) dst2[l] = (~r) & 1;
        }
        if (in.unit == Unit::Sfu) set_ready(w, in, now + cfg_.lat_sfu, kStallShortScoreboard);
        else set_ready(w, in, now + cfg_.lat_alu, kStallWait);
    }

    void exec_ldst(Sm& sm, Warp& w, const Instr& in, int pc, uint32_t exec, uint64_t now) {
        const bool is_load = in.op == Op::Ld;
        const Operand& addr_op = is_load ? in.ops[1] : in.ops[0];
        const Operand& val_op = is_load ? in.ops[0] : in.ops[1];
        const int esize = type_bytes(in.type);
        const int nvec = in.vec;
        WarpAccess acc;
        acc.mask = exec;
        acc.size = esize * nvec;
        Space space = in.space;
        for (int l = 0; l < 32; ++l) {
            if (!(exec >> l & 1)) continue;
            uint64_t a = address(w, addr_op, l);
            Space sp = resolve(in.space, a);
            space = sp;
            acc.addr[l] = a;
            for (int e = 0; e < nvec; ++e) {
                int reg = val_op.kind == Operand::Kind::Vec ? val_op.vec[e] : val_op.reg;
                uint64_t ea = a + (uint64_t)e * esize;
                if (is_load) {
                    uint64_t v = load_bytes(sm, w, sp, ea, esize, l, pc);
                    if (type_is_signed(in.type)) v = (uint64_t)sext(v, esize);
                    w.regs[(size_t)reg * 32 + l] = v;
                } else {
                    uint64_t v = val_op.kind == Operand::Kind::Imm ? (uint64_t)val_op.imm
                                                                   : w.regs[(size_t)reg * 32 + l];
                    store_bytes(sm, w, sp, ea, v, esize, l, pc);
                }
            }
        }
        if (exec == 0) { set_ready(w, in, now + cfg_.lat_alu, kStallWait); return; }
        if (space == Space::Param) {
            set_ready(w, in, now + cfg_.lat_param, kStallWait);
        } else if (space == Space::Shared) {
            uint64_t t = shared_timing(sm, acc, now);
            if (is_load) set_ready(w, in, t, kStallShortScoreboard);
        } else {
            uint64_t t = global_timing(sm, acc, !is_load, now);
            if (is_load) set_ready(w, in, t, kStallLongScoreboard);
        }
    }

    void exec_atomic(Sm& sm, Warp& w, const Instr& in, int pc, uint32_t exec, uint64_t now) {
        const bool has_dst = in.op == Op::Atom;
        const Operand& addr_op = has_dst ? in.ops[1] : in.ops[0];
        const Operand& b_op = has_dst ? in.ops[2] : in.ops[1];
        const Operand* c_op = has_dst && in.nops > 3 ? &in.ops[3] : nullptr;
        const int nb = type_bytes(in.type);
        const bool fl = type_is_float(in.type), sg = type_is_signed(in.type);
        WarpAccess acc;
        acc.mask = exec;
        acc.size = nb;
        Space space = in.space;
        uint64_t addrs[32];
        int n = 0;
        for (int l = 0; l < 32; ++l) {  // lanes are serialized in lane order
            if (!(exec >> l & 1)) continue;
            uint64_t a = address(w, addr_op, l);
            Space sp = resolve(in.space, a);
            space = sp;
            acc.addr[l] = a;
            addrs[n++] = a;
            uint64_t old = load_bytes(sm, w, sp, a, nb, l, pc);
            uint64_t b = read(sm, w, b_op, l);
            uint64_t nv = old;
            switch (in.aop) {
                case AtomOp::Add: nv = fl ? f32_bits(as_f32(old) + as_f32(b)) : old + b; break;
                case AtomOp::Min:
                    nv = fl ? f32_bits(std::fmin(as_f32(old), as_f32(b)))
                       : sg ? (sext(old, nb) < sext(b, nb) ? old : b) : std::min(old, trunc_to(b, nb));
                    break;
                case AtomOp::Max:
                    nv = fl ? f32_bits(std::fmax(as_f32(old), as_f32(b)))
                       : sg ? (sext(old, nb) > sext(b, nb) ? old : b) : std::max(old, trunc_to(b, nb));
                    break;
                case AtomOp::Exch: nv = b; break;
                case AtomOp::Cas: nv = trunc_to(old, nb) == trunc_to(b, nb) ? read(sm, w, *c_op, l) : old; break;
                case AtomOp::And: nv = old & b; break;
                case AtomOp::Or: nv = old | b; break;
                case AtomOp::Xor: nv = old ^ b; break;
            }
            store_bytes(sm, w, sp, a, trunc_to(nv, nb), nb, l, pc);
            if (has_dst && in.ops[0].kind == Operand::Kind::Reg) w.regs[(size_t)in.ops[0].reg * 32 + l] = old;
        }
        if (n == 0) { set_ready(w, in, now + cfg_.lat_alu, kStallWait); return; }
        // Same-address updates are serialized by the atomic unit.
        std::sort(addrs, addrs + n);
        int unique = 0, worst = 0, run = 0;
        for (int i = 0; i < n; ++i) {
            run = (i > 0 && addrs[i] == addrs[i - 1]) ? run + 1 : 1;
            if (run == 1) ++unique;
            worst = std::max(worst, run);
        }
        st_.atomic_requests++;
        st_.atomic_serialized += n - unique;
        uint64_t t;
        if (space == Space::Shared) {
            int degree = std::max(bank_conflict_degree(acc, cfg_.smem_banks), worst);
            st_.shared_requests++;
            st_.shared_wavefronts += degree;
            sm.lsu_free = now + degree;
            t = now + cfg_.lat_shared + 2 * degree;
            set_ready(w, in, t, kStallShortScoreboard);
        } else {
            CoalesceResult cr = coalesce(acc);
            sm.lsu_free = now + std::max(1, cr.lines);
            double start = std::max((double)now, l2_free_);
            for (uint64_t s : cr.sectors) {
                sm.l1.access(s * 32, true, false);
                auto r2 = l2_.access(s * 32, true);
                if (r2.hit) st_.l2_hits++;
                else { st_.l2_misses++; st_.dram_read_bytes += 32; dram_free_ = std::max(dram_free_, start) + 32.0 / cfg_.dram_bytes_per_cycle; }
                if (r2.dirty_evicted_bytes) st_.dram_write_bytes += r2.dirty_evicted_bytes;
            }
            l2_free_ = start + (double)worst + 32.0 * cr.sectors.size() / cfg_.l2_bytes_per_cycle;
            t = (uint64_t)start + cfg_.lat_atomic + worst;
            set_ready(w, in, t, kStallLongScoreboard);
        }
    }

    void exec_shfl(Sm& sm, Warp& w, const Instr& in, uint32_t exec, uint64_t now) {
        // shfl.sync.mode.b32 d[|p], a, b, c, membermask
        uint64_t src[32];
        for (int l = 0; l < 32; ++l) src[l] = read(sm, w, in.ops[1], l);
        uint64_t* dst = &w.regs[(size_t)in.ops[0].reg * 32];
        uint64_t* pdst = in.dst2 >= 0 ? &w.regs[(size_t)in.dst2 * 32] : nullptr;
        for (int l = 0; l < 32; ++l) {
            if (!(exec >> l & 1)) continue;
            uint32_t b = (uint32_t)read(sm, w, in.ops[2], l);
            uint32_t c = (uint32_t)read(sm, w, in.ops[3], l);
            int seg = (c >> 8) & 0x1f;
            int clamp = c & 0x1f;
            int min_lane = l & seg;
            int max_lane = min_lane | (clamp & ~seg);
            int j = 0;
            bool ok = false;
            switch (in.shfl) {
                case ShflMode::Up: j = l - (int)(b & 0x1f); ok = j >= max_lane; break;
                case ShflMode::Down: j = l + (int)(b & 0x1f); ok = j <= max_lane; break;
                case ShflMode::Bfly: j = l ^ (int)(b & 0x1f); ok = j <= max_lane; break;
                case ShflMode::Idx: j = min_lane | ((int)(b & 0x1f) & ~seg); ok = j <= max_lane; break;
            }
            if (!ok) j = l;
            dst[l] = trunc_to(src[j], 4);
            if (pdst) pdst[l] = ok ? 1 : 0;
        }
        sm.lsu_free = now + 1;
        set_ready(w, in, now + cfg_.lat_shared, kStallShortScoreboard);
    }

    // ---------------------------------------------------------------- control
    void maybe_release_barrier(Sm& sm, Cta& c) {
        if (c.arrived == 0 || c.arrived < c.live_warps) return;
        st_.barriers++;
        c.arrived = 0;
        for (int wi : c.warps) sm.warps[wi].at_barrier = false;
    }

    void reconverge(Sm& sm, int wi) {
        Warp& w = sm.warps[wi];
        while (!w.stack.empty()) {
            const SimtEntry& top = w.stack.back();
            if ((top.mask & ~w.exited) == 0 || top.pc == top.rpc) w.stack.pop_back();
            else break;
        }
        if (!w.stack.empty()) return;
        // Warp finished.
        w.active = false;
        if (w.at_barrier) { w.at_barrier = false; }
        Cta& c = sm.ctas[w.cta];
        c.live_warps--;
        sm.resident_warps--;
        if (c.live_warps == 0) {
            c.active = false;
            sm.resident_ctas--;
            done_ctas_++;
        } else {
            maybe_release_barrier(sm, c);
        }
    }
};

}  // namespace

Gpu::Gpu(GpuConfig cfg) : cfg_(cfg) {
    if (cfg_.smem_banks < 1 || cfg_.smem_banks > 64) throw std::invalid_argument("smem_banks must be 1..64");
    if (cfg_.num_sms < 1 || cfg_.schedulers_per_sm < 1) throw std::invalid_argument("bad SM configuration");
}
Gpu::~Gpu() = default;

KernelStats Gpu::launch(const Kernel& k, Dim3 grid, Dim3 block, const KernelArgs& args, const LaunchOptions& opt) {
    if (block.count() == 0 || block.count() > (uint32_t)cfg_.max_threads_per_cta)
        throw SimError("invalid block size for kernel " + k.name);
    if (grid.count() == 0) throw SimError("empty grid for kernel " + k.name);
    if (args.args().size() != k.params.size())
        throw SimError("kernel " + k.name + " expects " + std::to_string(k.params.size()) + " arguments, got " +
                       std::to_string(args.args().size()));
    std::vector<uint8_t> params(std::max(k.param_bytes, 1), 0);
    for (size_t i = 0; i < k.params.size(); ++i) {
        const auto& a = args.args()[i];
        if (a.size != k.params[i].size)
            throw SimError("argument " + std::to_string(i) + " (" + k.params[i].name + ") of " + k.name +
                           " has size " + std::to_string(a.size) + ", expected " + std::to_string(k.params[i].size));
        std::memcpy(params.data() + k.params[i].offset, &a.bits, a.size);
    }
    const int regs = opt.regs_per_thread > 0 ? opt.regs_per_thread : k.estimated_regs_per_thread();
    const int smem = opt.dynamic_shared_bytes > 0 ? dynamic_shared_base(k) + opt.dynamic_shared_bytes
                                                   : k.static_shared_bytes;
    KernelStats st;
    st.kernel = k.name;
    st.occupancy = compute_occupancy(cfg_, (int)block.count(), regs, smem);
    if (st.occupancy.ctas_per_sm == 0)
        throw SimError("kernel " + k.name + " cannot launch: a single CTA exceeds the SM's " + st.occupancy.limiter);
    Engine e(cfg_, mem_, k, grid, block, std::move(params), st.occupancy, smem, st);
    e.run();
    return st;
}

}  // namespace ptxsim
