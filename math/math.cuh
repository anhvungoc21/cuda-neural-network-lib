#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "./primitives/activation_functions.cuh"
#include "./primitives/loss_functions.cuh"

#define THRESHOLD_NUM_THREADS_USE_GPU 2048
#define THRESHOLD_NUM_OPS_USE_GPU 4096

// Performs a matrix multiplication
void matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                     size_t cols_A, size_t rows_B, size_t cols_B,
                     bool force_use_cpu);

// Applies an activation function to an array
void activate_arr(float *arr, size_t size, activation_func_t act_func);