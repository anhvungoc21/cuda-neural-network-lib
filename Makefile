# Compiler & flags
CC := nvcc
CFLAGS := -g

# File patterns to compile
CU_FILES := $(wildcard **/*.cu) $(wildcard math/primitives/*.cu)
CUH_FILES := $(wildcard **/*.cuh) $(wildcard math/primitives/*.cuh)

# Exclude archived experiments
CU_FILES := $(filter-out $(wildcard experiments/*), $(CU_FILES))

# Target executable
TARGET := main

all: main

main: main.cu $(CU_FILES) $(CUH_FILES)
	$(CC) $(CFLAGS) $(CU_FILES) -o $(TARGET) main.cu

format:
	@clang-format -i --style=file $(CU_FILES) $(CUH_FILES)

clean:
	rm -f main
	rm -rf *.dSYM

.PHONY: format clean all
