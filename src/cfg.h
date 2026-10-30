// Control-flow graph and immediate post-dominator analysis.
//
// SIMT hardware without independent thread scheduling reconverges divergent
// lanes at the immediate post-dominator (IPDOM) of the divergent branch. This
// module finds that point for every branch so the simulator's reconvergence
// stack knows where to merge masks again.
#pragma once

#include "ptx.h"

#include <vector>

namespace ptxsim {

struct BasicBlock {
    int start = 0;  // first instruction index
    int end = 0;    // one past the last instruction
    std::vector<int> succs;  // block ids; exit_block() for kernel exit
};

struct Cfg {
    std::vector<BasicBlock> blocks;
    std::vector<int> block_of;  // instruction index -> block id
    int exit_block() const { return (int)blocks.size(); }
};

Cfg build_cfg(const std::vector<Instr>& code);

// ipdom[b] for each block b; exit_block() when the IPDOM is the kernel exit.
std::vector<int> immediate_post_dominators(const Cfg& cfg);

// Per instruction: the pc where lanes split by that branch reconverge.
// Non-branch instructions get -1; code.size() means "at kernel exit".
std::vector<int> compute_reconvergence(const std::vector<Instr>& code);

}  // namespace ptxsim
