#include "cfg.h"
#include "ptx.h"
#include "test.h"

using namespace ptxsim;

static Kernel kernel_from_body(const std::string& body) {
    std::string src = ".visible .entry k(.param .u32 k_n)\n{\n .reg .pred %p<4>;\n .reg .b32 %r<8>;\n" + body + "}\n";
    return parse_ptx(src).kernel("k");
}

TEST(ipdom_of_if_else_diamond_is_the_join) {
    Kernel k = kernel_from_body(R"(
	mov.u32 %r1, %tid.x;            // 0
	setp.lt.u32 %p1, %r1, 16;       // 1
	@%p1 bra $L__then;              // 2
	mov.u32 %r2, 1;                 // 3  else
	bra.uni $L__join;               // 4
$L__then:
	mov.u32 %r2, 2;                 // 5
$L__join:
	add.s32 %r3, %r2, 1;            // 6
	ret;                            // 7
)");
    CHECK_EQ(k.reconverge_pc[2], 6);
    CHECK_EQ(k.reconverge_pc[4], 6);
}

TEST(ipdom_of_loop_exit_branch_is_after_the_loop) {
    Kernel k = kernel_from_body(R"(
	mov.u32 %r1, 0;                 // 0
$L__loop:
	add.s32 %r1, %r1, 1;            // 1
	setp.lt.u32 %p1, %r1, %tid.x;   // 2
	@%p1 bra $L__loop;              // 3
	mov.u32 %r2, %r1;               // 4
	ret;                            // 5
)");
    CHECK_EQ(k.reconverge_pc[3], 4);
}

TEST(ipdom_is_kernel_exit_when_a_path_returns_early) {
    Kernel k = kernel_from_body(R"(
	setp.lt.u32 %p1, %tid.x, 4;     // 0
	@%p1 bra $L__out;               // 1
	mov.u32 %r1, 1;                 // 2
	ret;                            // 3
$L__out:
	mov.u32 %r1, 2;                 // 4
	ret;                            // 5
)");
    CHECK_EQ(k.reconverge_pc[1], (int)k.code.size());
}

TEST(cfg_splits_blocks_at_branches_and_targets) {
    Kernel k = kernel_from_body(R"(
	setp.lt.u32 %p1, %tid.x, 4;
	@%p1 bra $L__b;
	mov.u32 %r1, 1;
$L__b:
	ret;
)");
    Cfg cfg = build_cfg(k.code);
    CHECK_EQ(cfg.blocks.size(), (size_t)3);
    CHECK_EQ(cfg.blocks[0].succs.size(), (size_t)2);
    CHECK_EQ(cfg.blocks[2].succs[0], cfg.exit_block());
}
