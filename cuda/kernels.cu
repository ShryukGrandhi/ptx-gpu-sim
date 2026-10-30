// CUDA C++ versions of the kernels in kernels/*.ptx.
//
// Names and parameter lists match the hand-written PTX, so the PTX that
// `nvcc -ptx` produces from this file can be loaded by the simulator and
// checked by the same test harness (see `make nvcc-conformance`).
// gpu_check.cu runs these kernels on a real GPU.

#define TILE 32
#define BLOCK_ROWS 8
#define MM_TILE 16

extern "C" __global__ void vecadd(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void strided_copy(const float* in, float* out, unsigned n, unsigned stride) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i * stride];
}

extern "C" __global__ void copy_vec4(const float4* in, float4* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 4 < n) out[i] = in[i];
}

extern "C" __global__ void reduce_interleaved(const float* in, float* out, unsigned n) {
    __shared__ float sdata[256];
    unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

extern "C" __global__ void reduce_strided_index(const float* in, float* out, unsigned n) {
    __shared__ float sdata[256];
    unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        unsigned index = 2 * s * tid;
        if (index < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

extern "C" __global__ void reduce_sequential(const float* in, float* out, unsigned n) {
    __shared__ float sdata[256];
    unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

extern "C" __global__ void reduce_warp_shuffle(const float* in, float* out, unsigned n) {
    __shared__ float partial[32];
    unsigned tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = warp_sum(i < n ? in[i] : 0.0f);
    if ((tid & 31) == 0) partial[tid >> 5] = v;
    __syncthreads();
    if ((tid >> 5) == 0) {
        v = warp_sum(tid < (blockDim.x + 31) / 32 ? partial[tid] : 0.0f);
        if (tid == 0) out[blockIdx.x] = v;
    }
}

extern "C" __global__ void transpose_naive(const float* in, float* out, unsigned n) {
    unsigned x = blockIdx.x * TILE + threadIdx.x;
    unsigned y = blockIdx.y * TILE + threadIdx.y;
    for (unsigned j = 0; j < TILE; j += BLOCK_ROWS) out[x * n + (y + j)] = in[(y + j) * n + x];
}

template <int PITCH>
__device__ __forceinline__ void transpose_tiled(const float* in, float* out, unsigned n, float* tile) {
    unsigned x = blockIdx.x * TILE + threadIdx.x;
    unsigned y = blockIdx.y * TILE + threadIdx.y;
    for (unsigned j = 0; j < TILE; j += BLOCK_ROWS) tile[(threadIdx.y + j) * PITCH + threadIdx.x] = in[(y + j) * n + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (unsigned j = 0; j < TILE; j += BLOCK_ROWS) out[(y + j) * n + x] = tile[threadIdx.x * PITCH + threadIdx.y + j];
}

extern "C" __global__ void transpose_smem(const float* in, float* out, unsigned n) {
    __shared__ float tile[TILE * TILE];
    transpose_tiled<TILE>(in, out, n, tile);
}

extern "C" __global__ void transpose_smem_padded(const float* in, float* out, unsigned n) {
    __shared__ float tile[TILE * (TILE + 1)];
    transpose_tiled<TILE + 1>(in, out, n, tile);
}

extern "C" __global__ void matmul_naive(const float* a, const float* b, float* c, unsigned n) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float acc = 0.0f;
    for (unsigned k = 0; k < n; ++k) acc += a[row * n + k] * b[k * n + col];
    c[row * n + col] = acc;
}

extern "C" __global__ void matmul_tiled(const float* a, const float* b, float* c, unsigned n) {
    __shared__ float as[MM_TILE][MM_TILE];
    __shared__ float bs[MM_TILE][MM_TILE];
    unsigned tx = threadIdx.x, ty = threadIdx.y;
    unsigned row = blockIdx.y * MM_TILE + ty, col = blockIdx.x * MM_TILE + tx;
    float acc = 0.0f;
    for (unsigned t = 0; t < n; t += MM_TILE) {
        as[ty][tx] = a[row * n + t + tx];
        bs[ty][tx] = b[(t + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < MM_TILE; ++k) acc += as[ty][k] * bs[k][tx];
        __syncthreads();
    }
    c[row * n + col] = acc;
}

extern "C" __global__ void histogram_global(const unsigned* in, unsigned* bins, unsigned n) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&bins[in[i] & 255u], 1u);
}

extern "C" __global__ void histogram_shared(const unsigned* in, unsigned* bins, unsigned n) {
    __shared__ unsigned local[256];
    for (unsigned b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&local[in[i] & 255u], 1u);
    __syncthreads();
    for (unsigned b = threadIdx.x; b < 256; b += blockDim.x) atomicAdd(&bins[b], local[b]);
}
