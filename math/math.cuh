#include <stdio.h>
#include <stdlib.h>

#define THRESHOLD_USE_GPU 8192

// Performs a matrix multiplication. Uses the GPU if specified, otherwise
// automatically decided based on the size of the input
void matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                     size_t cols_A, size_t rows_B, size_t cols_B, bool force_use_gpu);