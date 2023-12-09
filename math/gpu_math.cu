#include "../utils/errors.cuh"

#include "gpu_math.cuh"

/**
 * Kernel for matrix multiplication
 *
 * @param A First matrix (m x p)
 * @param B Second matrix (p x n)
 * @param result Result matrix (m x n)
 */
__global__ void __kernel_matrix_multiply(float *A, float *B, float *result,
                                         size_t rows_A, size_t inner_dim,
                                         size_t cols_B) {
  // Get row and column of thread in result matrix
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check boundaries
  if (row >= rows_A || col >= cols_B)
    return;

  // Calculate cell
  float sum = 0.0f;
  for (int k = 0; k < inner_dim; k++) {
    sum += A[row * inner_dim + k] * B[k * cols_B + col];
  }

  // Store result
  result[row * cols_B + col] = sum;
}

/**
 * Performs a matrix multiplication on the GPU
 *
 * @param A First matrix (m x p)
 * @param B Second matrix (p x n)
 * @param result Result matrix (m x n)
 */
void gpu__matrix_multiply(float *A, float *B, float *result, size_t rows_A,
                          size_t inner_dim, size_t cols_B) {
  // Set up on GPU
  float *gpu_A, *gpu_B, *gpu_result;
  size_t size_A = sizeof(float) * (rows_A * inner_dim);
  size_t size_B = sizeof(float) * (inner_dim * cols_B);
  size_t size_result = sizeof(float) * (rows_A * cols_B);

  gpuErrchk(cudaMalloc(&gpu_A, size_A));
  gpuErrchk(cudaMalloc(&gpu_B, size_B));
  gpuErrchk(cudaMalloc(&gpu_result, size_result));

  gpuErrchk(cudaMemcpy(gpu_A, A, size_A, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(gpu_B, B, size_B, cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(gpu_result, result, size_result, cudaMemcpyHostToDevice));

  // Each block has a fixed 32 x 32 threads
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

  // Each grid has a X and Y corresponding to the shape of result matrix
  // The plus one extra unit and - 1 is to ensure rounding up
  size_t gridCols = (cols_B + blockSize.x - 1) / blockSize.x;
  size_t gridRows = (rows_A + blockSize.y - 1) / blockSize.y;
  dim3 gridSize(gridCols, gridRows);

  // Run kernel
  __kernel_matrix_multiply<<<gridSize, blockSize>>>(gpu_A, gpu_B, gpu_result,
                                                    rows_A, inner_dim, cols_B);

  // Copy result back to CPU
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(
      cudaMemcpy(result, gpu_result, size_result, cudaMemcpyDeviceToHost));

  // Clean up
  cudaFree(gpu_A);
  cudaFree(gpu_B);
  cudaFree(gpu_result);
}
