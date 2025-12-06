NVCC = nvcc
CUDA_PATH = /usr/local/cuda

CUDA_ARCH ?= detect

ifeq ($(CUDA_ARCH),detect)
    DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')
    ifneq ($(DETECTED_ARCH),)
        CUDA_ARCH_FLAG = -arch=sm_$(DETECTED_ARCH)
    else
        CUDA_ARCH_FLAG = -arch=sm_50 -gencode=arch=compute_50,code=sm_50 \
                         -gencode=arch=compute_60,code=sm_60 \
                         -gencode=arch=compute_70,code=sm_70 \
                         -gencode=arch=compute_75,code=sm_75 \
                         -gencode=arch=compute_80,code=sm_80 \
                         -gencode=arch=compute_86,code=sm_86 \
                         -gencode=arch=compute_89,code=sm_89 \
                         -gencode=arch=compute_90,code=sm_90
    endif
else
    CUDA_ARCH_FLAG = -arch=sm_$(CUDA_ARCH)
endif

NVCC_FLAGS = $(CUDA_ARCH_FLAG) \
             -O3 \
             -use_fast_math \
             -lineinfo \
             --expt-relaxed-constexpr \
             -Xcompiler -fopenmp \
             -Xcompiler -march=native \
             -Xptxas -v \
             --default-stream per-thread

INCLUDES = -I$(CUDA_PATH)/include -I.
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand

TARGET = forest_fire_sim
SOURCES = main.cu forest_fire_simulation.cu
HEADERS = forest_fire.cuh

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)

profile: $(TARGET)
	nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency,shared_efficiency,sm_efficiency ./$(TARGET)

nsys_profile: $(TARGET)
	nsys profile --stats=true --force-overwrite=true -o forest_fire ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.ppm *.nsys-rep *.sqlite

run: $(TARGET)
	./$(TARGET)

debug: NVCC_FLAGS += -G -g
debug: $(TARGET)
	cuda-gdb ./$(TARGET)

memcheck: $(TARGET)
	cuda-memcheck ./$(TARGET)

info:
	@echo "Detected GPU Compute Capability: $(DETECTED_ARCH)"
	@echo "Using CUDA Architecture Flag: $(CUDA_ARCH_FLAG)"

.PHONY: all clean run profile nsys_profile debug memcheck info