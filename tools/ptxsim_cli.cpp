// ptxsim info <file.ptx> [block_threads]
//
// Static view of each kernel in a PTX file: resources, the basic blocks of
// its control-flow graph, where each branch reconverges, and the occupancy
// the default simulated GPU would allow.
#include "cfg.h"
#include "gpu.h"
#include "ptx.h"

#include <cstdlib>
#include <iostream>

using namespace ptxsim;

static int info(const std::string& path, int block) {
    Module m = load_ptx_file(path);
    GpuConfig cfg;
    for (const Kernel& k : m.kernels) {
        std::cout << "kernel " << k.name << "\n";
        std::cout << "  params:";
        for (const auto& p : k.params) std::cout << " " << p.name << "@" << p.offset << "(" << p.size << "B)";
        std::cout << "\n  instructions: " << k.code.size() << ", static shared: " << k.static_shared_bytes
                  << " B, estimated registers/thread: " << k.estimated_regs_per_thread() << "\n";
        Cfg g = build_cfg(k.code);
        std::cout << "  basic blocks: " << g.blocks.size() << "\n";
        for (size_t b = 0; b < g.blocks.size(); ++b) {
            std::cout << "    B" << b << " [" << g.blocks[b].start << ", " << g.blocks[b].end << ") ->";
            for (int s : g.blocks[b].succs) {
                if (s == g.exit_block()) std::cout << " exit";
                else std::cout << " B" << s;
            }
            std::cout << "\n";
        }
        for (size_t i = 0; i < k.code.size(); ++i) {
            const Instr& in = k.code[i];
            if (in.op != Op::Bra || in.guard < 0) continue;
            int r = k.reconverge_pc[i];
            std::cout << "  branch pc " << i << " (line " << in.line << ") reconverges at "
                      << (r == (int)k.code.size() ? std::string("exit") : "pc " + std::to_string(r)) << "\n";
        }
        Occupancy o = compute_occupancy(cfg, block, k.estimated_regs_per_thread(), k.static_shared_bytes);
        std::cout << "  occupancy @" << block << " threads/CTA: " << o.ctas_per_sm << " CTAs/SM, "
                  << o.active_warps_per_sm << "/" << cfg.max_warps_per_sm << " warps (" << int(o.theoretical * 100)
                  << "%), limited by " << o.limiter << "\n\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "info") {
        std::cerr << "usage: ptxsim info <file.ptx> [block_threads]\n";
        return 2;
    }
    try {
        return info(argv[2], argc > 3 ? std::atoi(argv[3]) : 256);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
