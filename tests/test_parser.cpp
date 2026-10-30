#include "ptx.h"
#include "test.h"

using namespace ptxsim;

static const char* kSmall = R"(
.version 7.0
.target sm_70
.address_size 64

.visible .entry small(
	.param .u32 small_param_n,
	.param .u64 small_param_ptr,
	.param .align 4 .b8 small_param_blob[12]
)
.maxntid 128, 1, 1
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<4>;
	.reg .f32 	%f<3>;
	.reg .b64 	%rd<3>;
	.shared .align 4 .b8 buf[64];
	.shared .align 8 .f32 tail[3];

	ld.param.u32 	%r1, [small_param_n];      // comment
	ld.param.u64 	%rd1, [small_param_ptr];
	mov.f32 	%f1, 0f3F800000;  /* 1.0 */
	mov.u32 	%r2, buf;
	mov.u32 	%r3, tail;
	setp.lt.s32 	%p1, %r1, -5;
	@!%p1 bra 	$L__end;
	ld.global.v2.f32 	{%f1, %f2}, [%rd1+-8];
$L__end:
	ret;
}
)";

TEST(parser_reads_params_with_abi_alignment) {
    Module m = parse_ptx(kSmall);
    CHECK_EQ(m.kernels.size(), (size_t)1);
    const Kernel& k = m.kernel("small");
    CHECK_EQ(k.params.size(), (size_t)3);
    CHECK_EQ(k.params[0].offset, 0);
    CHECK_EQ(k.params[1].offset, 8);  // u64 is 8-byte aligned
    CHECK_EQ(k.params[2].offset, 16);
    CHECK_EQ(k.params[2].size, 12);
    CHECK_EQ(k.param_bytes, 28);
}

TEST(parser_expands_register_ranges_and_shared_symbols) {
    Module m = parse_ptx(kSmall);
    const Kernel& k = m.kernel("small");
    CHECK(k.reg_index.count("%r0"));
    CHECK(k.reg_index.count("%r3"));
    CHECK(!k.reg_index.count("%r4"));
    CHECK_EQ(k.shared_symbols.at("buf"), 0);
    CHECK_EQ(k.shared_symbols.at("tail"), 64);
    CHECK_EQ(k.static_shared_bytes, 76);
    // 4 b32 + 3 f32 + 3 b64 (2 slots each); predicates are not counted
    CHECK_EQ(k.estimated_regs_per_thread(), 13);
}

TEST(parser_decodes_operands) {
    Module m = parse_ptx(kSmall);
    const Kernel& k = m.kernel("small");
    const Instr& mov = k.code[2];
    CHECK(mov.op == Op::Mov);
    CHECK(mov.ops[1].kind == Operand::Kind::Imm);
    CHECK_EQ(mov.ops[1].imm, (int64_t)0x3F800000);
    CHECK_EQ(k.code[4].ops[1].imm, 64);  // `tail` resolved to its shared offset
    const Instr& setp = k.code[5];
    CHECK(setp.op == Op::Setp && setp.cmp == Cmp::Lt && setp.type == Type::S32);
    CHECK_EQ(setp.ops[2].imm, -5);
    const Instr& bra = k.code[6];
    CHECK(bra.op == Op::Bra && bra.guard_neg && bra.target == 8);
    const Instr& ld = k.code[7];
    CHECK(ld.vec == 2 && ld.ops[0].kind == Operand::Kind::Vec && ld.ops[0].nvec == 2);
    CHECK_EQ(ld.ops[1].imm, -8);
    CHECK(ld.unit == Unit::MemGlobal);
    CHECK_EQ(ld.defs.size(), (size_t)2);
}

TEST(parser_reports_unsupported_instruction_with_line) {
    const char* src = ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry k()\n{\n  .reg .f32 %fd<2>;\n"
                      "  tex.1d.v4.f32.s32 %fd1, [t, {%fd0}];\n  ret;\n}\n";
    CHECK_THROWS(parse_ptx(src), ParseError, "line 7");
    CHECK_THROWS(parse_ptx(src), ParseError, "unsupported instruction");
}

TEST(parser_rejects_undefined_label_and_register) {
    const char* bad_label = ".visible .entry k()\n{\n bra.uni $L__nowhere;\n}\n";
    CHECK_THROWS(parse_ptx(bad_label), ParseError, "undefined label");
    const char* bad_reg = ".visible .entry k()\n{\n .reg .b32 %r<2>;\n add.s32 %r1, %r7, 1;\n}\n";
    CHECK_THROWS(parse_ptx(bad_reg), ParseError, "unknown operand %r7");
}

TEST(parser_appends_implicit_ret) {
    const char* src = ".visible .entry k()\n{\n .reg .b32 %r<2>;\n mov.u32 %r1, 3;\n}\n";
    Module m = parse_ptx(src);
    const Kernel& k = m.kernel("k");
    CHECK_EQ(k.code.size(), (size_t)2);
    CHECK(k.code.back().op == Op::Ret);
}

TEST(parser_loads_all_repository_kernels) {
    const char* files[] = {"memory_access.ptx", "reduction.ptx", "transpose.ptx", "matmul.ptx", "histogram.ptx"};
    int kernels = 0;
    for (const char* f : files) kernels += (int)load_ptx_file(testing::kernel_dir() + "/" + f).kernels.size();
    CHECK_EQ(kernels, 14);
}
