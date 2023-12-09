#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Performs a matrix multiplication on the CPU
void cpu__matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                          size_t inner_dim, size_t cols_B);
