# ptx-gpu-sim

A cycle-level SIMT GPU simulator, written in C++17, that runs **PTX** kernels. You can feed it
hand-written PTX or the PTX that `nvcc -ptx` produces from CUDA C++. It executes every thread,
so results can be checked against a CPU reference. While running, it models the GPU behaviour that
decides how fast a CUDA kernel is: warp scheduling, branch divergence and reconvergence, memory
coalescing, shared-memory bank conflicts, L1/L2/DRAM traffic, atomics, barriers and occupancy.

The repository includes 14 kernels in classic before/after pairs (naive vs tiled SGEMM, three
transposes, four reductions, global vs privatized histograms, strided and vectorized copies). For
each kernel the benchmark reports the hardware-level reason one version beats another, e.g.
32 sectors per request vs 4, 16.5 shared-memory wavefronts per request vs 1, or 78% vs 96% SIMD
efficiency.

I wrote it to learn how NVIDIA GPUs execute CUDA code below the source level, without access to a
GPU. **Simulated cycle counts are model outputs, not predictions for any real GPU.** The model is
not calibrated against hardware (see [Limitations](#limitations)).

## What it does

- **PTX front end** (`src/ptx_parser.cpp`). Parses `.entry` kernels with `.param` lists (ABI
  alignment), `.reg` ranges, static `.shared` and `extern .shared` (dynamic) arrays, labels,
  predicated instructions, vector operands (`.v2`/`.v4`) and `[reg+imm]` addresses.
  It covers about 40 opcodes: integer/float ALU, `mad`/`fma`, `mul.wide/hi`, `setp`/`selp`,
  `cvt`/`cvta`, SFU math, `ld`/`st` for global, shared, param and generic memory,
  `atom`/`red`, `shfl.sync` (up/down/bfly/idx with segment masks), `bar.sync`, `bra`, `ret`.
  Unsupported PTX is rejected with a line number instead of being skipped silently.
- **Reconvergence analysis** (`src/cfg.cpp`). Builds the control-flow graph and computes
  immediate post-dominators. Every warp has a SIMT reconvergence stack: a divergent branch pushes one
  entry per path, and the paths merge again at the branch's IPDOM.
- **Execution engine** (`src/gpu.cpp`). CTAs are dispatched to SMs up to the occupancy limit,
  which is set by warps, CTA slots, registers or shared memory. Each SM has several warp schedulers
  (greedy-then-oldest or loose round-robin), each issuing at most one instruction per cycle. A
  per-register scoreboard stalls dependent instructions until their inputs arrive. Every idle
  scheduler cycle is charged to one stall reason, using Nsight Compute's names:
  `long_scoreboard`, `short_scoreboard`, `wait`, `barrier`, `mem_throttle`, `idle`.
- **Memory system** (`src/memory.cpp`):
  - Each warp-wide global access is split into unique 32-byte sectors.
  - L1 is a sectored, set-associative, write-through LRU cache, one per SM. L2 is a shared
    write-back cache.
  - L2 and DRAM sit behind bandwidth queues.
  - Shared-memory accesses are split into wavefronts by 32-bank conflict analysis, with broadcast
    when lanes read the same word.
  - Atomics on the same address are serialized.
  - Out-of-bounds global or shared accesses raise an error that names the CTA, thread, PTX line and
    instruction.
- **CUDA sources** (`cuda/kernels.cu`). CUDA C++ versions of every kernel, with the same names and
  parameters as the PTX. CI compiles them with `nvcc -ptx` and runs the generated PTX through the
  simulator with the same correctness checks. `cuda/gpu_check.cu` runs them on a real GPU and times
  them with CUDA events.

## Running it

Requirements: a C++17 compiler and `make`. The CUDA toolkit is optional and needed only for the
`ptx-check`, `nvcc-conformance` and `cuda` targets. Those targets do not need a GPU.

```sh
make                     # builds build/ptxsim_tests, build/ptxsim_bench, build/ptxsim
make test                # 56 tests; the 4 nvcc conformance tests skip without nvcc output
make bench               # full benchmark (the tables below)
make bench-quick         # smaller sizes, under a second

# with the CUDA toolkit (or inside the nvidia/cuda devel container):
make ptx-check           # assemble kernels/*.ptx with ptxas for sm_80
make nvcc-conformance    # nvcc -ptx cuda/kernels.cu, then run that PTX in the simulator
make cuda                # build cuda/gpu_check.cu (running it needs an NVIDIA GPU)

./build/ptxsim info kernels/reduction.ptx 256   # CFG, reconvergence points, occupancy
```

Example `ptxsim info` output (first kernel):

```
kernel reduce_interleaved
  params: reduce_interleaved_param_in@0(8B) reduce_interleaved_param_out@8(8B) reduce_interleaved_param_n@16(4B)
  instructions: 43, static shared: 1024 B, estimated registers/thread: 37
  basic blocks: 10
    B0 [0, 12) -> B1 B2
    B1 [12, 15) -> B2
    B2 [15, 21) -> B3
    B3 [21, 23) -> B4 B7
    B4 [23, 27) -> B5 B6
    B5 [27, 33) -> B6
    B6 [33, 36) -> B3
    B7 [36, 38) -> B8 B9
    B8 [38, 42) -> B9
    B9 [42, 43) -> exit
  branch pc 11 (line 36) reconverges at pc 15
  branch pc 22 (line 49) reconverges at pc 36
  branch pc 26 (line 53) reconverges at pc 33
  branch pc 37 (line 66) reconverges at pc 42
  occupancy @256 threads/CTA: 6 CTAs/SM, 48/48 warps (100%), limited by warps
```

Using the library from C++:

```cpp
ptxsim::Module m = ptxsim::load_ptx_file("kernels/matmul.ptx");
ptxsim::Gpu gpu;                                   // default GpuConfig, see src/gpu.h
uint64_t a = gpu.upload(host_a), b = gpu.upload(host_b), c = gpu.alloc_zeroed<float>(n * n);
ptxsim::KernelStats s = gpu.launch(m.kernel("matmul_tiled"), {n / 16, n / 16}, {16, 16},
                                   ptxsim::KernelArgs().ptr(a).ptr(b).ptr(c).u32(n));
std::cout << s.report();                           // cycles, IPC, sectors/request, stalls, ...
auto result = gpu.download<float>(c, n * n);
```

## Architecture

```mermaid
flowchart LR
    subgraph Front end
        PTX[".ptx file<br/>(hand-written or nvcc -ptx)"] --> P[ptx_parser<br/>tokens → Instr]
        P --> C[cfg<br/>basic blocks, IPDOM]
        C --> K[Kernel<br/>code + reconvergence pcs]
    end
    subgraph "Gpu::launch"
        K --> D[CTA dispatcher<br/>occupancy limits]
        D --> SM1[SM 0]
        D --> SMn[SM N-1]
    end
    subgraph "SM (x num_sms)"
        direction TB
        WS[warp schedulers x4<br/>GTO / LRR] --> SB[scoreboard<br/>per-register ready cycle]
        SB --> EX[execute at issue<br/>32 lanes, SIMT stack]
        EX --> LSU[LSU<br/>coalescer / bank model]
        LSU --> L1[(L1 sectored<br/>write-through)]
    end
    SM1 -.-> WS
    L1 --> L2[(L2 shared<br/>write-back, BW queue)]
    L2 --> DRAM[(DRAM<br/>latency + BW queue)]
    EX --> GM[(GlobalMemory<br/>functional data)]
```

How one instruction is processed:

1. A scheduler looks at its warps. A warp is ready if it is not waiting at a barrier, every source
   and destination register is ready (scoreboard), and, for memory instructions, the SM's load/store
   unit is free.
2. The scheduler issues the chosen warp's instruction. The engine executes it functionally for the
   active lanes (the top-of-stack mask, minus exited lanes, ANDed with the guard predicate). Values
   are therefore always exact. Timing only decides when they become visible.
3. Each kind of instruction sets the result's ready time a different way:
   - ALU: fixed latency.
   - Shared memory: base latency, plus one LSU cycle per wavefront from bank conflicts.
   - Global memory: the warp's addresses are coalesced into sectors. Each sector is looked up in
     L1, then L2, then DRAM, where it waits in the bandwidth queue. The load is ready when its
     slowest sector arrives, and the LSU stays busy for one cycle per 128-byte line touched.
4. Branches update the warp's reconvergence stack. If the active lanes split, the current entry's
   pc jumps to the IPDOM and one entry is pushed for each path. An entry is popped when it reaches
   its reconvergence pc or when all its lanes have exited.
5. When no scheduler on an SM can issue, the SM sleeps until the earliest scoreboard or LSU wake-up
   time. The engine skips over these idle cycles instead of stepping through them one by one, and
   still charges each skipped cycle to the right stall reason.

| Module | Role |
|---|---|
| `src/ptx.h`, `src/ptx_parser.cpp` | PTX data model and parser |
| `src/cfg.h`, `src/cfg.cpp` | CFG, post-dominators, reconvergence pcs |
| `src/memory.h`, `src/memory.cpp` | device memory, coalescer, bank conflicts, sectored cache |
| `src/gpu.h`, `src/gpu.cpp` | config, occupancy, scheduler, SIMT stack, timing, statistics |
| `kernels/*.ptx` | 14 hand-written kernels (valid PTX: CI assembles them with `ptxas`) |
| `cuda/kernels.cu`, `cuda/gpu_check.cu` | the same kernels in CUDA C++, plus a real-GPU runner |
| `tests/` | 60 tests (parser, CFG, memory model, SIMT semantics, kernels, nvcc conformance) |
| `bench/bench.cpp` | benchmark that prints the tables below |
| `tools/ptxsim_cli.cpp` | `ptxsim info` |

## Benchmark results

Produced by `make bench`, which runs `./build/ptxsim_bench kernels`. The program checks every run
against a CPU reference and exits non-zero if any result is wrong. All runs below passed. All
values are simulated cycles and counters from the default `GpuConfig`, so the same build always
produces the same simulated values. CI runs the same benchmark on every push.

Simulated GPU: 8 SMs x 4 schedulers, 48 warps/SM, 32 KiB L1/SM, 1024 KiB L2, DRAM 64 B/cycle, latencies L1/L2/DRAM 32/190/400 cycles.

#### Global memory coalescing: strided_copy, n = 262144

| stride | sectors/request | DRAM bytes read | cycles | useful B/cycle (load + store) | slowdown |
|---|---:|---:|---:|---:|---:|
| 1 | 4.00 | 1048576 | 24278 | 86.38 | 1.00x |
| 2 | 8.00 | 2097152 | 43456 | 48.26 | 1.79x |
| 4 | 16.00 | 4194304 | 78294 | 26.79 | 3.22x |
| 8 | 32.00 | 8388608 | 145284 | 14.43 | 5.98x |
| 32 | 32.00 | 8388608 | 146660 | 14.30 | 6.04x |

#### Vectorized access: scalar vs float4 copy, n = 262144

| kernel | warp instructions | load requests | sectors/request | cycles |
|---|---:|---:|---:|---:|
| strided_copy (stride 1) | 163840 | 8192 | 4.00 | 24278 |
| copy_vec4 | 36864 | 2048 | 16.00 | 22005 |

#### Block reduction variants, n = 1048576, 256 threads/CTA

| kernel | cycles | speedup | SIMD efficiency | divergent branches | shared requests | bank-conflict wavefronts | warp instructions |
|---|---:|---:|---:|---:|---:|---:|---:|
| reduce_interleaved | 154882 | 1.00x | 77.8% | 196608 | 614400 | 0 | 4382720 |
| reduce_strided_index | 144657 | 1.07x | 95.9% | 24576 | 184320 | 430080 | 3620864 |
| reduce_sequential | 125234 | 1.24x | 96.2% | 24576 | 184320 | 0 | 2998272 |
| reduce_warp_shuffle | 99357 | 1.56x | 93.1% | 40960 | 36864 | 0 | 1675264 |

#### Matrix transpose, 512 x 512 floats

| kernel | cycles | speedup | load sectors/req | store sectors/req | shared wavefronts/req | mem_throttle stalls |
|---|---:|---:|---:|---:|---:|---:|
| transpose_naive | 37724 | 1.00x | 4.00 | 32.00 | 0.00 | 16.1% |
| transpose_smem | 50142 | 0.75x | 4.00 | 4.00 | 16.50 | 31.8% |
| transpose_smem_padded | 24305 | 1.55x | 4.00 | 4.00 | 1.00 | 0.2% |

#### SGEMM, 256 x 256, 16x16 CTAs

| kernel | cycles | speedup | FLOP/cycle | global load requests | L1 hit rate | DRAM bytes read | IPC | long_scoreboard stalls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul_naive | 290007 | 1.00x | 115.70 | 1048576 | 87.2% | 524288 | 14.69 | 39.4% |
| matmul_tiled | 210128 | 1.38x | 159.69 | 65536 | 23.1% | 524288 | 22.54 | 12.8% |

#### Histogram (256 bins), n = 262144, 64 CTAs x 256 threads

| kernel | input | cycles | warp atomic requests | serialized lane updates |
|---|---:|---:|---:|---:|
| histogram_global | 256 bins (uniform) | 40991 | 8192 | 15244 |
| histogram_shared | 256 bins (uniform) | 19756 | 8704 | 15244 |
| histogram_global | 4 bins (skewed) | 94414 | 8192 | 229378 |
| histogram_shared | 4 bins (skewed) | 20480 | 8704 | 229378 |

#### Occupancy vs latency hiding: vecadd, n = 262144, 128 threads/CTA, CTAs/SM capped with dynamic shared memory

| CTAs/SM | theoretical occupancy | achieved occupancy | cycles | kernel B/cycle (2 loads + 1 store) | long_scoreboard stalls |
|---|---:|---:|---:|---:|---:|
| 1 | 8.3% | 8.2% | 117526 | 26.77 | 88.9% |
| 2 | 16.7% | 16.3% | 59062 | 53.26 | 88.5% |
| 4 | 33.3% | 32.5% | 43770 | 71.87 | 85.2% |
| 8 | 66.7% | 65.0% | 43460 | 72.38 | 85.1% |
| 12 | 100.0% | 96.9% | 43460 | 72.38 | 85.1% |

#### Warp scheduling policy: greedy-then-oldest vs loose round-robin

| workload | GTO cycles | LRR cycles | GTO L1 hit rate | LRR L1 hit rate |
|---|---:|---:|---:|---:|
| matmul_naive 256 | 290007 | 264514 | 87.2% | 88.0% |
| matmul_tiled 256 | 210128 | 215978 | 23.1% | 33.0% |
| transpose_smem_padded 512 | 24305 | 24941 | 0.0% | 0.0% |
| vecadd 262144 | 43460 | 43898 | 0.0% | 0.0% |

#### Simulator throughput (host)

Measured on an Apple M3 Ultra (macOS, Apple clang 21, `-O2`, single-threaded). This is the only
table that depends on the host machine:

| simulations | simulated cycles | warp instructions | thread instructions | host seconds | warp instr / s | thread instr / s |
|---|---:|---:|---:|---:|---:|---:|
| 33 | 3220841 | 43510784 | 1349005312 | 13.93 | 3.12 M | 96.84 M |

Host time includes generating inputs and running the CPU reference checks.

### Reading the results

- **Coalescing.** Sectors per request go 4 → 8 → 16 → 32 as the stride grows, and cycles go up by
  about 6x. Once a warp touches one sector per lane (stride ≥ 8), DRAM moves 8x more bytes than the
  kernel actually uses.
- **Reductions.** The interleaved kernel (`tid % (2*s)`) has the most divergent branches and the
  lowest SIMD efficiency (77.8%). Strided indexing removes most of the divergence but adds 430,080
  extra shared-memory wavefronts from bank conflicts. Sequential addressing removes both. Warp
  shuffles cut shared-memory requests by 5x and give a 1.56x speedup over the first version.
- **Transpose.** The 32x32 tile makes the global stores coalesced (32 → 4 sectors per request). But
  reading a tile column costs 16.5 wavefronts per shared request (32 for the column reads, 1 for
  the row writes), so in this model it is *slower* than the naive kernel. Padding the tile to
  32x33 brings it down to 1.00 wavefront per request and gives the best time (1.55x). Real GPUs
  usually rank the 32x32 tiled version above the naive one, partly because naive partial-sector
  writes cost real DRAM bandwidth. Here the whole 512x512 matrix fits in the 1 MiB L2, which absorbs
  those writes.
- **SGEMM.** Tiling cuts global load requests 16x and `long_scoreboard` stalls from 39.4% to 12.8%.
  At n = 256 both versions read the same 512 KiB from DRAM, because the naive kernel's reuse is
  already caught by L1 (87.2% hit rate). The 1.38x gain comes from fewer memory instructions and
  shorter stalls, not from lower DRAM traffic.
- **Histogram.** When the input uses only 4 bins, 229,378 lane updates have to be serialized at the
  atomic unit, and the global-atomic version slows down 2.3x. The shared-memory privatized version
  barely changes.
- **Occupancy.** Going from 1 to 4 resident CTAs per SM hides memory latency and speeds up vecadd
  2.7x. Beyond that the kernel is limited by DRAM bandwidth (configured at 64 B/cycle; reads alone
  run at about 48 B/cycle), so more occupancy does not help.

## Tests

`make test` runs 60 tests (4 of them skip unless nvcc-generated PTX is supplied). They cover:

- **Parser:** ABI parameter offsets, register ranges, operand decoding, and errors with line numbers.
- **CFG:** IPDOM for if/else diamonds, loops, and early exits.
- **Memory model:** coalescing for contiguous, misaligned, strided and vector accesses; bank
  conflict degrees for strides 1/2/4/32/33 and broadcast; cache LRU, sector fills, write-back of
  dirty sectors and write-through invalidation.
- **Occupancy:** each of the four limiters.
- **SIMT semantics:**
  - nested divergence, per-lane loop trip counts and early `ret`
  - barriers across warps, including warps that have already exited
  - all four shuffle modes and segmented shuffles
  - atomics returning unique old values
  - integer and float edge cases (arithmetic shift, signed `rem`/`div`, `mul.hi`, `mul.wide`,
    `cvt` truncation and sign extension)
  - 2D indexing and dynamic shared memory
  - out-of-bounds detection
  - the cycle-limit guard
- **Timing:** a dependent ALU chain pays the latency and independent instructions do not; extra
  warps hide latency.
- **Kernels:** the result of every kernel, plus the architectural effect each variant is meant to
  show (asserted, not just printed), and that every scheduler cycle is accounted for.
- **nvcc conformance:** runs the same kernel checks on PTX that nvcc generated from `cuda/kernels.cu`.

CI (`.github/workflows/ci.yml`) has two jobs:

1. Builds with g++ and clang++, runs the tests and both benchmark sizes.
2. Runs inside the `nvidia/cuda:12.6.3-devel` container. It assembles every hand-written kernel
   with `ptxas`, compiles the CUDA sources with `nvcc`, and runs the nvcc-generated PTX through the
   simulator.

## Limitations

- **Not calibrated against real hardware.** Latencies, bandwidths and cache sizes in `GpuConfig`
  are round numbers in a plausible range. Use the simulator to compare kernel variants and explain
  them, not to predict milliseconds on a particular GPU. `cuda/gpu_check.cu` exists so the rankings
  can be compared with a real GPU, but I have not run it: I had no NVIDIA GPU, and CI only compiles
  it.
- **Pre-Volta reconvergence.** Divergence uses an IPDOM stack. It does not model Volta+ independent
  thread scheduling, so code that relies on intra-warp synchronization between divergent paths
  (e.g. spin locks between lanes) behaves differently.
- **Execution at issue.** Instructions take effect when they issue, so data races between warps are
  not reproduced. Memory fences are no-ops.
- **Simplified pipeline.** The model does not include:
  - an instruction cache, dual issue, or separate pipelines per functional unit (one ALU latency,
    one SFU latency)
  - register bank conflicts or operand collectors
  - constant or texture caches (param loads take a fixed latency)
  - DRAM channels, banks or row buffers (one GPU-wide bandwidth queue)
  - the L1/shared-memory carve-out
- **Register estimate.** Occupancy uses the number of declared PTX virtual registers, which
  overstates what `ptxas` actually allocates. For example, `reduce_interleaved` declares 37 but
  `ptxas` uses 12. The `LaunchOptions::regs_per_thread` setting overrides this.
- **PTX coverage.** Only 32-bit float and integer types up to 64 bits are supported. There is no
  f64/f16, textures, `.local`/`.const` memory, device function calls, tensor-core (`mma`/`wmma`)
  instructions, or cooperative groups beyond `bar.sync`. The parser rejects these with an error.
- **Speed.** About 3 M warp-instructions per second on one host core. Large grids take seconds to
  minutes.
