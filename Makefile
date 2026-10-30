CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
BUILD    := build

LIB_SRCS := src/ptx_parser.cpp src/cfg.cpp src/memory.cpp src/gpu.cpp
LIB_OBJS := $(LIB_SRCS:src/%.cpp=$(BUILD)/%.o)
TEST_SRCS := $(wildcard tests/*.cpp)
TEST_OBJS := $(TEST_SRCS:tests/%.cpp=$(BUILD)/tests/%.o)

.PHONY: all test bench bench-quick ptx-check nvcc-conformance cuda clean

all: $(BUILD)/ptxsim_tests $(BUILD)/ptxsim_bench $(BUILD)/ptxsim

$(BUILD)/%.o: src/%.cpp src/*.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/tests/%.o: tests/%.cpp tests/*.h src/*.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Isrc -Itests -c $< -o $@

$(BUILD)/libptxsim.a: $(LIB_OBJS)
	ar rcs $@ $^

$(BUILD)/ptxsim_tests: $(TEST_OBJS) $(BUILD)/libptxsim.a
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/ptxsim_bench: bench/bench.cpp tests/kernel_harness.h $(BUILD)/libptxsim.a
	$(CXX) $(CXXFLAGS) -Isrc -Itests $< $(BUILD)/libptxsim.a -o $@

$(BUILD)/ptxsim: tools/ptxsim_cli.cpp $(BUILD)/libptxsim.a
	$(CXX) $(CXXFLAGS) -Isrc $< $(BUILD)/libptxsim.a -o $@

test: $(BUILD)/ptxsim_tests
	./$(BUILD)/ptxsim_tests kernels

bench: $(BUILD)/ptxsim_bench
	./$(BUILD)/ptxsim_bench kernels

bench-quick: $(BUILD)/ptxsim_bench
	./$(BUILD)/ptxsim_bench kernels --quick

# The targets below need the CUDA toolkit (nvcc/ptxas); no GPU is required.
NVCC ?= nvcc
PTXAS ?= ptxas
CUDA_ARCH ?= sm_80

# Assemble every hand-written kernel with NVIDIA's assembler to prove it is valid PTX.
ptx-check:
	@for f in kernels/*.ptx; do echo "ptxas $$f"; $(PTXAS) -arch=$(CUDA_ARCH) -v $$f -o /dev/null || exit 1; done

# Compile the CUDA reference kernels to PTX with nvcc and run them through the simulator.
nvcc-conformance: $(BUILD)/ptxsim_tests
	@mkdir -p $(BUILD)/nvcc
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -ptx cuda/kernels.cu -o $(BUILD)/nvcc/kernels_nvcc.ptx
	./$(BUILD)/ptxsim_tests kernels --nvcc-ptx $(BUILD)/nvcc/kernels_nvcc.ptx --filter nvcc

# Build the real-GPU checker (running it needs an NVIDIA GPU).
cuda:
	@mkdir -p $(BUILD)
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -o $(BUILD)/gpu_check cuda/gpu_check.cu cuda/kernels.cu

clean:
	rm -rf $(BUILD)
