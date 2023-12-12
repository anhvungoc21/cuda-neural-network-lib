#pragma once

#include <stdlib.h>

#include "./primitives/activation_functions.cuh"

// Performs a matrix multiplication on the CPU
void cpu__matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                          size_t inner_dim, size_t cols_B);

// Applies a specified activation on an array on the CPU
void cpu__activate_arr(float *arr, size_t size, activation_func_t acti_func);