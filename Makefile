# Compiler & flags
CC := nvcc
CFLAGS := -g

# File patterns to compile
C_FILES := $(wildcard **/*.c)
H_FILES := $(wildcard **/*.h)
CU_FILES := $(wildcard **/*.cu)
CUH_FILES := $(wildcard **/*.cuh)

# Exclude archived experiments
CU_FILES := $(filter-out $(wildcard experiments/*), $(CU_FILES))

# Target executable
TARGET := main

all: main

main: main.cu $(C_FILES) $(H_FILES) $(CU_FILES) $(CUH_FILES)
	$(CC) $(CFLAGS) $(C_FILES) $(CU_FILES) -o $(TARGET) main.cu

format:
	@clang-format -i --style=file $(C_FILES) $(H_FILES) $(CU_FILES) $(CUH_FILES)

clean:
	rm -f main
	rm -rf *.dSYM

.PHONY: format clean all
