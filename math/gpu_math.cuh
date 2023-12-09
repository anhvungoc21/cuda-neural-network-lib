#pragma once

#include <stdlib.h>

// Size of a thread block
// Fixed to 32 so that each block has 32 x 32 = 1024 threads
#define BLOCK_SIZE 32

// Performs a matrix multiplication on the GPU
void gpu__matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                          size_t inner_dim, size_t cols_B);