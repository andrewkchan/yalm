MAKEFLAGS+=-r -j

UNAME=$(shell uname)

NVCC?=nvcc

BUILD=build
ASM_DIR=$(BUILD)/asm

# compile .c, .cpp, .cu files
SOURCES=$(filter-out src/test.cpp,$(wildcard src/*.c))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cc))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cpp))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cu))
SOURCES+=$(wildcard vendor/*.c)
SOURCES+=$(wildcard vendor/*.cc)
SOURCES+=$(wildcard vendor/*.cpp)
SOURCES+=$(wildcard vendor/*.cu)

# Define test sources separately
TEST_SOURCES=src/test.cpp
TEST_SOURCES+=$(filter-out src/main.cpp,$(SOURCES))

OBJECTS=$(SOURCES:%=$(BUILD)/%.o)
TEST_OBJECTS=$(TEST_SOURCES:%=$(BUILD)/%.o)
ASM_FILES=$(patsubst %.cpp,$(ASM_DIR)/%.s,$(filter %.cpp,$(SOURCES)))
TEST_ASM_FILES=$(patsubst %.cpp,$(ASM_DIR)/%.s,$(filter %.cpp,$(TEST_SOURCES)))

BINARY=$(BUILD)/main
TEST_BINARY=$(BUILD)/test

CFLAGS=-g -Wall -Wpointer-arith -Werror -O3 -ffast-math -Ivendor -std=c++20
LDFLAGS=-lm

CFLAGS+=-fopenmp -mf16c -mavx2 -mfma
LDFLAGS+=-fopenmp
LDFLAGS+=-lcudart

ifneq (,$(wildcard /usr/local/cuda))
  LDFLAGS+=-L/usr/local/cuda/lib64
endif

CUFLAGS+=-O2 -lineinfo -Ivendor
CUFLAGS+=-allow-unsupported-compiler # for recent CUDA versions

ifeq ($(CUARCH),)
  CUFLAGS+=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 --threads 2
else
  CUFLAGS+=-arch=$(CUARCH)
endif

all: $(BINARY) asm

test: $(TEST_BINARY) test-asm

# Target to build just assembly files
asm: $(ASM_FILES)

test-asm: $(TEST_ASM_FILES)

format:
	clang-format -i src/*

$(BINARY): $(OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

$(TEST_BINARY): $(TEST_OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

# Rule to generate assembly for cpp files
$(ASM_DIR)/%.s: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -S -masm=intel -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cc.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)
-include $(TEST_OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD)

.PHONY: all clean format test asm test-asm