#!/usr/bin/env python3
"""Generates kernels/transpose.ptx.

The two shared-memory transposes differ only in the tile row pitch (32 vs 33
floats), so they come from one template to keep everything else identical.
Usage: python3 tools/gen_transpose_ptx.py kernels/transpose.ptx
"""
import sys

HEADER = """//
// Matrix transpose of an n x n float matrix (n % 32 == 0), 32x32 tiles,
// blockDim = (32, 8): each thread moves 4 elements.
//   transpose_naive        coalesced reads, uncoalesced (column) writes
//   transpose_smem         staged through a 32x32 shared tile: coalesced both ways,
//                          but reading a tile column is a 32-way bank conflict
//   transpose_smem_padded  32x33 tile: the extra column puts each row in a different bank
//
// Generated from tools/gen_transpose_ptx.py; edit the generator, not this file.
//
.version 7.0
.target sm_70
.address_size 64

.visible .entry transpose_naive(
	.param .u64 transpose_naive_param_in,
	.param .u64 transpose_naive_param_out,
	.param .u32 transpose_naive_param_n
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<16>;
	.reg .f32 	%f<2>;
	.reg .b64 	%rd<9>;

	ld.param.u64 	%rd1, [transpose_naive_param_in];
	ld.param.u64 	%rd2, [transpose_naive_param_out];
	ld.param.u32 	%r1, [transpose_naive_param_n];
	cvta.to.global.u64 	%rd3, %rd1;
	cvta.to.global.u64 	%rd4, %rd2;
	mov.u32 	%r2, %tid.x;
	mov.u32 	%r3, %tid.y;
	mov.u32 	%r4, %ctaid.x;
	mov.u32 	%r5, %ctaid.y;
	shl.b32 	%r6, %r4, 5;
	add.s32 	%r7, %r6, %r2;
	shl.b32 	%r8, %r5, 5;
	add.s32 	%r9, %r8, %r3;
	mov.u32 	%r10, 0;
$L__loop:
	add.s32 	%r11, %r9, %r10;
	mad.lo.s32 	%r12, %r11, %r1, %r7;
	mul.wide.u32 	%rd5, %r12, 4;
	add.s64 	%rd6, %rd3, %rd5;
	ld.global.f32 	%f1, [%rd6];
	mad.lo.s32 	%r13, %r7, %r1, %r11;
	mul.wide.u32 	%rd7, %r13, 4;
	add.s64 	%rd8, %rd4, %rd7;
	st.global.f32 	[%rd8], %f1;
	add.s32 	%r10, %r10, 8;
	setp.lt.u32 	%p1, %r10, 32;
	@%p1 bra 	$L__loop;
	ret;
}

"""

TEMPLATE = """.visible .entry {name}(
	.param .u64 {name}_param_in,
	.param .u64 {name}_param_out,
	.param .u32 {name}_param_n
)
{{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<20>;
	.reg .f32 	%f<2>;
	.reg .b64 	%rd<9>;
	.shared .align 4 .b8 {name}_tile[{bytes}];

	ld.param.u64 	%rd1, [{name}_param_in];
	ld.param.u64 	%rd2, [{name}_param_out];
	ld.param.u32 	%r1, [{name}_param_n];
	cvta.to.global.u64 	%rd3, %rd1;
	cvta.to.global.u64 	%rd4, %rd2;
	mov.u32 	%r2, %tid.x;
	mov.u32 	%r3, %tid.y;
	mov.u32 	%r4, %ctaid.x;
	mov.u32 	%r5, %ctaid.y;
	shl.b32 	%r6, %r4, 5;
	add.s32 	%r7, %r6, %r2;
	shl.b32 	%r8, %r5, 5;
	add.s32 	%r9, %r8, %r3;
	mov.u32 	%r10, {name}_tile;
	mov.u32 	%r11, 0;
$L__load:
	add.s32 	%r12, %r9, %r11;
	mad.lo.s32 	%r13, %r12, %r1, %r7;
	mul.wide.u32 	%rd5, %r13, 4;
	add.s64 	%rd6, %rd3, %rd5;
	ld.global.f32 	%f1, [%rd6];
	add.s32 	%r14, %r3, %r11;
	mad.lo.s32 	%r15, %r14, {pitch}, %r2;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r17, %r10, %r16;
	st.shared.f32 	[%r17], %f1;
	add.s32 	%r11, %r11, 8;
	setp.lt.u32 	%p1, %r11, 32;
	@%p1 bra 	$L__load;
	bar.sync 	0;
	add.s32 	%r18, %r8, %r2;
	add.s32 	%r19, %r6, %r3;
	mov.u32 	%r11, 0;
$L__store:
	add.s32 	%r14, %r3, %r11;
	mad.lo.s32 	%r15, %r2, {pitch}, %r14;
	shl.b32 	%r16, %r15, 2;
	add.s32 	%r17, %r10, %r16;
	ld.shared.f32 	%f1, [%r17];
	add.s32 	%r12, %r19, %r11;
	mad.lo.s32 	%r13, %r12, %r1, %r18;
	mul.wide.u32 	%rd5, %r13, 4;
	add.s64 	%rd6, %rd4, %rd5;
	st.global.f32 	[%rd6], %f1;
	add.s32 	%r11, %r11, 8;
	setp.lt.u32 	%p1, %r11, 32;
	@%p1 bra 	$L__store;
	ret;
}}

"""

def main():
    out = HEADER
    out += TEMPLATE.format(name="transpose_smem", pitch=32, bytes=32 * 32 * 4)
    out += TEMPLATE.format(name="transpose_smem_padded", pitch=33, bytes=32 * 33 * 4)
    with open(sys.argv[1], "w") as f:
        f.write(out)

if __name__ == "__main__":
    main()
