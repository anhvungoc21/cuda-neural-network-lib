#include "math.cuh"

#include "gpu_math.cuh"

extern "C" {
#include "cpu_math.h"
}

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
  if (force_use_cpu || num_threads < THRESHOLD_USE_GPU) {
    cpu__matrix_multiply(A, B, result, rows_A, cols_A, cols_B);
  } else {
    gpu__matrix_multiply(A, B, result, rows_A, cols_A, cols_B);
  }
}