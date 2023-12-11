#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "primitives/activation_functions.h"

#define THRESHOLD_USE_GPU 8192

// Performs a matrix multiplication
void matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                     size_t cols_A, size_t rows_B, size_t cols_B,
                     bool force_use_cpu);

// Applies an activation function to an array
void activate(float *arr, activation_func_t act_func, bool force_use_cpu);