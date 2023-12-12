#include "math.cuh"

#include "cpu_math.cuh"
#include "gpu_math.cuh"

#include "./primitives/activation_functions.cuh"

/**
 * Performs a matrix multiplication.
 * Uses the CPU/GPU based on the size of the input.
 *
 * @param A First matrix (m x p)
 * @param B Second matrix (p x n)
 * @param result Result matrix (m x n)
 */
void matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                     size_t cols_A, size_t rows_B, size_t cols_B,
                     bool force_use_cpu) {
  // Guard against invalid matrix inputs
  if (cols_A != rows_B) {
    fprintf(stderr, "Unabled to multiply %dx%d matrix by %dx%d matrix", rows_A,
            cols_A, rows_B, cols_B);
    return;
  }

  // Select GPU or CPU
  size_t num_threads = rows_A * cols_B;
  size_t num_ops_per_thread = cols_A * rows_B;
  if (force_use_cpu || ((num_threads < THRESHOLD_NUM_THREADS_USE_GPU) && (num_ops_per_thread < THRESHOLD_NUM_OPS_USE_GPU))) {
    cpu__matrix_multiply(A, B, result, rows_A, cols_A, cols_B);
  } else {
    gpu__matrix_multiply(A, B, result, rows_A, cols_A, cols_B);
  }
}

/**
 * Applies a specific activation function on an array
 * TODO later on: Implement GPU version
 */
void activate_arr(float *arr, size_t size, activation_func_t acti_func) {
  cpu__activate_arr(arr, size, acti_func);
}


/**
 * Calculates the total loss function from an array
 */ 
float calculate_loss(float *predicted, float *actual, size_t size, loss_func_t loss_func) {
  return cpu__calculate_loss(predicted, actual, size, loss_func);
}

/**
 * Calculates the derivative of a loss function on the CPU
 */ 
float derivative_loss_func(float predicted, float actual, loss_func_t loss_func) {
  return cpu__derivative_loss_func(predicted, actual, loss_func);   
}

/**
 * Calculates the derivative of an activation function on the CPU
 */ 
float derivative_acti_func(float output, activation_func_t acti_func) {
  return cpu__derivative_acti_func(output, acti_func);   
}