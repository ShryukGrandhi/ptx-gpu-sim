#include "cfg.h"

#include <algorithm>
#include <set>

namespace ptxsim {

static bool is_terminator(const Instr& in) { return in.op == Op::Bra || in.op == Op::Ret || in.op == Op::Exit; }

Cfg build_cfg(const std::vector<Instr>& code) {
    Cfg cfg;
    const int n = (int)code.size();
    std::set<int> leaders = {0};
    for (int i = 0; i < n; ++i) {
        if (code[i].op == Op::Bra && code[i].target >= 0) leaders.insert(code[i].target);
        if (is_terminator(code[i]) && i + 1 < n) leaders.insert(i + 1);
    }
    std::vector<int> starts(leaders.begin(), leaders.end());
    cfg.block_of.assign(n, -1);
    for (size_t b = 0; b < starts.size(); ++b) {
        BasicBlock bb;
        bb.start = starts[b];
        bb.end = b + 1 < starts.size() ? starts[b + 1] : n;
        for (int i = bb.start; i < bb.end; ++i) cfg.block_of[i] = (int)b;
        cfg.blocks.push_back(bb);
    }
    const int exit = cfg.exit_block();
    for (auto& bb : cfg.blocks) {
        const Instr& last = code[bb.end - 1];
        bool guarded = last.guard >= 0;
        auto fallthrough = [&]() { return bb.end < n ? cfg.block_of[bb.end] : exit; };
        if (last.op == Op::Bra) {
            bb.succs.push_back(cfg.block_of[last.target]);
            if (guarded) bb.succs.push_back(fallthrough());
        } else if (last.op == Op::Ret || last.op == Op::Exit) {
            bb.succs.push_back(exit);
            if (guarded) bb.succs.push_back(fallthrough());
        } else {
            bb.succs.push_back(fallthrough());
        }
        std::sort(bb.succs.begin(), bb.succs.end());
        bb.succs.erase(std::unique(bb.succs.begin(), bb.succs.end()), bb.succs.end());
    }
    return cfg;
}

std::vector<int> immediate_post_dominators(const Cfg& cfg) {
    // Iterative data-flow: pdom(exit) = {exit}; pdom(b) = {b} U intersect(pdom(s) for s in succs(b)).
    const int nb = (int)cfg.blocks.size();
    const int total = nb + 1;
    const int exit = cfg.exit_block();
    std::vector<std::vector<bool>> pdom(total, std::vector<bool>(total, true));
    pdom[exit].assign(total, false);
    pdom[exit][exit] = true;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int b = nb - 1; b >= 0; --b) {
            std::vector<bool> nset(total, true);
            for (int s : cfg.blocks[b].succs)
                for (int i = 0; i < total; ++i) nset[i] = nset[i] && pdom[s][i];
            nset[b] = true;
            if (nset != pdom[b]) { pdom[b] = std::move(nset); changed = true; }
        }
    }
    // The immediate post-dominator is the strict post-dominator that every other
    // strict post-dominator also post-dominates (the "closest" one).
    std::vector<int> ipdom(nb, exit);
    for (int b = 0; b < nb; ++b) {
        std::vector<int> strict;
        for (int i = 0; i < total; ++i)
            if (i != b && pdom[b][i]) strict.push_back(i);
        for (int c : strict) {
            bool closest = true;
            for (int o : strict)
                if (o != c && !pdom[c][o]) { closest = false; break; }
            if (closest) { ipdom[b] = c; break; }
        }
    }
    return ipdom;
}

std::vector<int> compute_reconvergence(const std::vector<Instr>& code) {
    std::vector<int> rpc(code.size(), -1);
    if (code.empty()) return rpc;
    Cfg cfg = build_cfg(code);
    std::vector<int> ipdom = immediate_post_dominators(cfg);
    for (size_t i = 0; i < code.size(); ++i) {
        if (code[i].op != Op::Bra && code[i].op != Op::Ret && code[i].op != Op::Exit) continue;
        int b = cfg.block_of[i];
        int p = ipdom[b];
        rpc[i] = p == cfg.exit_block() ? (int)code.size() : cfg.blocks[p].start;
    }
    return rpc;
}

}  // namespace ptxsim
