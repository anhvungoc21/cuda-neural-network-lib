CC := nvcc
CFLAGS := -g

all: main

main: main.cu math/cpu_math.c math/cpu_math.h math/gpu_math.cu math/gpu_math.cuh utils/utils.c utils/utils.h utils/errors.cu utils/errors.cuh
	$(CC) $(CFLAGS) -o main main.cu math/gpu_math.cu utils/errors.cu math/cpu_math.c utils/utils.c

format:
	@clang-format -i --style=file $(wildcard */*.c) $(wildcard */*.h) \
								  $(wildcard */*.cu) $(wildcard */*.cuh)

clean:
	rm -f main
	rm -rf *.dSYM

.PHONY: format clean all
