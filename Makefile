CC := nvcc
CFLAGS := -g

all: main

main: main.cu math/cpu_math.c math/cpu_math.h math/gpu_math.cu math/gpu_math.cuh utils.c utils.h
	$(CC) $(CFLAGS) -o main main.cu

format:
	@clang-format -i --style=file $(wildcard */*.c) $(wildcard */*.h) \
								  $(wildcard */*.cu) $(wildcard */*.cuh)

clean:
	rm -f math.exe
	rm -rf *.dSYM

.PHONY: format clean all