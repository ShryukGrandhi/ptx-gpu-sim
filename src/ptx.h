// PTX subset: data model produced by the parser and consumed by the simulator.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ptxsim {

struct ParseError : std::runtime_error {
    ParseError(int line, const std::string& msg)
        : std::runtime_error("line " + std::to_string(line) + ": " + msg), line(line) {}
    int line;
};

enum class Type : uint8_t { None, Pred, U8, S8, U16, S16, U32, S32, U64, S64, B8, B16, B32, B64, F32 };

int type_bytes(Type t);
bool type_is_signed(Type t);
bool type_is_float(Type t);

enum class Op : uint8_t {
    Nop, Ld, St, Mov, Cvta, Cvt, Add, Sub, Mul, Mad, Fma, Div, Rem, Min, Max, Neg, Abs,
    And, Or, Xor, Not, Shl, Shr, Setp, Selp, Sqrt, Rsqrt, Rcp, Ex2, Lg2,
    Bra, Ret, Exit, Bar, Shfl, Atom, Red,
};

enum class Space : uint8_t { Generic, Global, Shared, Param };
enum class Cmp : uint8_t { Eq, Ne, Lt, Le, Gt, Ge };
enum class MulMode : uint8_t { Lo, Hi, Wide };
enum class ShflMode : uint8_t { Up, Down, Bfly, Idx };
enum class AtomOp : uint8_t { Add, Min, Max, Exch, Cas, And, Or, Xor };

// Functional-unit class, used by the timing model.
enum class Unit : uint8_t { Alu, Sfu, Branch, Barrier, MemParam, MemGlobal, MemShared, MemGeneric, Atomic };

enum class Special : uint8_t { TidX, TidY, TidZ, NtidX, NtidY, NtidZ, CtaidX, CtaidY, CtaidZ,
                               NctaidX, NctaidY, NctaidZ, LaneId, WarpId };

struct Operand {
    enum class Kind : uint8_t { None, Reg, Imm, Special, Addr, Vec };
    Kind kind = Kind::None;
    int reg = -1;           // Reg, or Addr base register (-1 for absolute address)
    int64_t imm = 0;        // Imm bits, or Addr byte offset
    Special special = Special::TidX;
    int vec[4] = {-1, -1, -1, -1};
    int nvec = 0;
};

struct Instr {
    Op op = Op::Nop;
    Unit unit = Unit::Alu;
    Type type = Type::None;   // primary type (dst type for cvt)
    Type type2 = Type::None;  // source type for cvt
    Space space = Space::Generic;
    Cmp cmp = Cmp::Eq;
    MulMode mul = MulMode::Lo;
    ShflMode shfl = ShflMode::Idx;
    AtomOp aop = AtomOp::Add;
    bool cvta_to = false;     // cvta.to.<space> vs cvta.<space>
    bool cmp_unsigned = false; // setp.lo/ls/hi/hs
    uint8_t vec = 1;          // .v2 / .v4
    int guard = -1;           // predicate register or -1
    bool guard_neg = false;
    int nops = 0;
    Operand ops[5];
    int dst2 = -1;            // second destination of setp/shfl (`%p|%q`)
    int target = -1;          // branch target instruction index
    std::vector<int> uses;    // registers read (scoreboard)
    std::vector<int> defs;    // registers written (scoreboard)
    int line = 0;
    std::string text;
};

struct Param {
    std::string name;
    int size = 0;
    int align = 1;
    int offset = 0;
};

struct RegInfo {
    std::string name;
    Type type;
};

struct Kernel {
    std::string name;
    std::vector<Param> params;
    int param_bytes = 0;
    std::vector<RegInfo> regs;
    std::unordered_map<std::string, int> reg_index;
    std::vector<Instr> code;
    int static_shared_bytes = 0;
    bool uses_dynamic_shared = false;
    std::unordered_map<std::string, int> shared_symbols;  // name -> byte offset
    std::vector<int> reconverge_pc;  // per instruction: IPDOM pc for branches (code.size() = exit)

    // Estimated 32-bit registers per thread (virtual PTX regs; ptxas may allocate fewer).
    int estimated_regs_per_thread() const;
};

struct Module {
    std::vector<Kernel> kernels;
    const Kernel& kernel(const std::string& name) const;
};

// Byte offset of `extern .shared` (dynamic) memory within a CTA's shared memory.
inline int dynamic_shared_base(const Kernel& k) { return (k.static_shared_bytes + 15) / 16 * 16; }

Module parse_ptx(const std::string& source);
Module load_ptx_file(const std::string& path);

}  // namespace ptxsim
