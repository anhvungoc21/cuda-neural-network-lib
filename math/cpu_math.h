#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Performs a matrix multiplication on the CPU
bool cpu__matrix_multiply(float *A, float *B, float *result, int rows_A,
                          int cols_A, int rows_B, int cols_B);