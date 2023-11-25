#include "gpu_math.cu"

#define BLOCK_SIZE 32

/**
 * ThKernel for matrix multiplication
 * 
 * @param A First matrix (m x p)
 * @param B Second matrix (p x n)
 * @param result Result matrix (m x n)
 * @param rows_A m
 * @param inner_dim p
 * @param cols_B n
 */
__global__ void __kernel_matrix_multiply(float *A, float *B, float *result,
                                         int rows_A, int inner_dim,
                                         int cols_B) {
  // Get row and column of thread in result matrix
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check boundaries
  if (row >= rows_A || col >= cols_B)
    return;

  // Calculate cell
  float sum = 0.0f;
  for (int k = 0; k < inner_dim; k++) {
    sum += A[row * inner_dim + k] + B[k * cols_B + col];
  }
  C[row * cols_B + col] = sum;
}

/**
 * Performs a matrix multiplication on the GPU
 *
 * @param A First matrix (m x p)
 * @param B Second matrix (p x n)
 * @param result Result matrix (m x n)
 * @return int Success (1) or Failure (0)
 */
bool gpu__matrix_multiply(float *A, float *B, float *result, int rows_A,
                          int cols_A, int rows_B, int cols_B) {
  // Guard against invalid matrix inputs
  if (cols_A != rows_B) {
    printf("Unabled to multiply %dx%d matrix by %dx%d matrix", rows_A, cols_A,
    rows_B, cols_B);
    return false;
  }

  // Set up on GPU
  float *gpu_A, *gpu_B, *gpu_result;
  size_t size_A = sizeof(float) * (rows_A * cols_A);
  size_t size_B = sizeof(float) * (rows_B * cols_B);
  size_t size_result = sizeof(float) * (rows_A * cols_B);

  cudaMalloc(&gpu_A, size_A);
  cudaMalloc(&gpu_B, size_B);
  cudaMalloc(&gpu_result, size_result);

  cudaMemcpy(gpu_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_result, result, size_result, cudaMemcpyHostToDevice);

  // Each block has a fixed 32 x 32 threads
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

  // Each grid has a X and Y corresponding to the shape of result matrix
  // The plus one extra unit and - 1 is to ensure rounding up
  size_t gridCols = (cols_B + blockSize.x - 1) / blockSize.x;
  size_t gridRows = (row_A + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(gridCols, gridRows);

  // Run kernel
  __kernel_matrix_multiply<<<gridSize, blockSize>>>(
      gpu_A, gpu_B, gpu_result, rows_A, cols_A, cols_B);

  // Copy result back to CPU
  cudaMemcpy(result, gpu_result, size_result, cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(gpu_A);
  cudaFree(gpu_B);
  cudaFree(gpu_result);

  return true;
}