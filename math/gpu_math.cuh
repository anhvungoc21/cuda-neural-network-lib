#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Performs a matrix multiplication on the GPU
bool gpu__matrix_multiply(float *A, float *B, float *result, int rows_A,
                          int cols_A, int rows_B, int cols_B);