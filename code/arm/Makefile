all: gemm conv sparse

BIN_DIR?=bin
EIGEN_PATH?=/usr/local/eigen
MKDIR=mkdir -p
ARM_COMPUTE_INCLUDE_PATH?=/usr/local/ARMComputeLib
ARM_COMPUTE_LIB_PATH?=/usr/local/ARMComputeLib/
ARM_COMPUTE_LIB=$(ARM_COMPUTE_LIB_PATH)/lib/linux-arm64-v8a-neon
KERNELS_DIR=../kernels/

gemm: gemm_bench.cc
	$(MKDIR) $(BIN_DIR)
#gcc
	g++ -O3 --std=c++11 -I/home/ubuntu/ -mcpu=cortex-a53 -mfpu=neon-vfpv4 -I $(KERNELS_DIR) -o bin/gemm_bench -lpthread $< -lpthread
#clang
#	clang-10 -O3 --std=c++11 -lstdc++ -mcpu=cortex-a53 -mfpu=neon-vfpv4 -lm -I/home/ubuntu/ -I $(KERNELS_DIR) -o bin/gemm_bench $< -lpthread

sparse: sparse_bench.cpp
	$(MKDIR) $(BIN_DIR)
	g++ -O3 -std=c++11  -mfpu=neon-fp-armv8 -o bin/sparse_bench $< \
		-I $(EIGEN_PATH) -I $(KERNELS_DIR)

conv: convolution.cpp
	$(MKDIR) $(BIN_DIR)
	g++ -O3 -std=c++11  -I $(ARM_COMPUTE_INCLUDE_PATH) -I $(KERNELS_DIR) -std=c++11 -larm_compute -L $(ARM_COMPUTE_LIB) $< -o bin/conv_bench

clean:
	rm -rf $(BIN_DIR)
