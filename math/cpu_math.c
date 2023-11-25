#include "cpu_math.h"

/**
 * Performs a matrix multiplication on the CPU
 *
 * @param A First matrix (m x p)
 * @param B Second matrix (p x n)
 * @param result Result matrix (m x n)
 * @return int Success (1) or Failure (0)
 */
bool cpu__matrix_multiply(float *A, float *B, float *result, int rows_A,
                          int cols_A, int rows_B, int cols_B) {
  // Check invalidity
  if (cols_A != rows_B) {
    printf("Unabled to multiply %dx%d matrix by %dx%d matrix", rows_A, cols_A,
           rows_B, cols_B);
    return false;
  }

  // Select row in A
  for (int r = 0; r < rows_A; r++) {
    // Select col in B
    for (int c = 0; c < cols_B; c++) {
      // Combine each index in selected row and col
      float sum = 0.0f;
      for (int k = 0; k < cols_A; k++) {
        sum += A[r * cols_A + k] * B[k * cols_B + c];
      }

      // Store cell result
      result[r * cols_B + c] = sum;
    }
  }

  return true;
}
