#include "cpu_math.h"

/**
 * Performs a matrix multiplication on the CPU
 *
 * @param A First matrix (m x p)
 * @param B Second matrix (p x n)
 * @param result Result matrix (m x n)
 */
void cpu__matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                          size_t inner_dim, size_t cols_B) {
  // Select row in A
  for (int r = 0; r < rows_A; r++) {
    // Select col in B
    for (int c = 0; c < cols_B; c++) {
      // Combine each index in selected row and col
      float sum = 0.0f;
      for (int k = 0; k < inner_dim; k++) {
        sum += A[r * inner_dim + k] * B[k * cols_B + c];
      }

      // Store cell result
      result[r * cols_B + c] = sum;
    }
  }
}
