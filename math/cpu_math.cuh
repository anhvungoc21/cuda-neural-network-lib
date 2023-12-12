#pragma once

#include <stdlib.h>

#include "./primitives/activation_functions.cuh"
#include "./primitives/loss_functions.cuh"

// Performs a matrix multiplication on the CPU
void cpu__matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                          size_t inner_dim, size_t cols_B);

// Applies a specified activation on an array on the CPU
void cpu__activate_arr(float *arr, size_t size, activation_func_t acti_func);


// Calculates a loss function for an array on the CPU 
float cpu__calculate_loss(float *predicted, float *actual, size_t size, loss_func_t loss_func);

// Calculates the derivative of a loss function on the CPU
float cpu__derivative_loss_func(float predicted, float actual, loss_func_t loss_func);

// Calculates the derivative of an activation function on the CPU
float cpu__derivative_acti_func(float output, activation_func acti_func);