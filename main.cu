#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "math/gpu_math.cuh"

extern "C" {
#include "math/cpu_math.h"
#include "utils/utils.h"
}


int main() {
  srand(time(NULL));

  int rows_A = 1024;
  int cols_A = 512;
  int rows_B = 512;
  int cols_B = 2048;

  float *A = (float *) malloc(sizeof(float) * (rows_A * cols_A));
  float *B = (float *) malloc(sizeof(float) * (rows_B * cols_B));
  float *cpu_result = (float *) malloc(sizeof(float) * (rows_A * cols_B));
  float *gpu_result = (float *) malloc(sizeof(float) * (rows_A * cols_B));

  fill_random_floats(A, rows_A * cols_A);
  fill_random_floats(B, rows_B * cols_B);

  // CPU
  clock_t start_cpu = clock();
  cpu__matrix_multiply(A, B, cpu_result, rows_A, cols_A, rows_B, cols_B);
  clock_t end_cpu = clock();
  printf("CPU: %.3f\n", (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC);

  // GPU
  clock_t start_gpu = clock();
  gpu__matrix_multiply(A, B, gpu_result, rows_A, cols_A, rows_B, cols_B);
  clock_t end_gpu = clock();
  printf("GPU: %.3f\n", (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC);

  // Check for result equality
  float tolerance = 1e-6;
  check_equal_arr(cpu_result, gpu_result, rows_A * cols_B, tolerance);

  free(A);
  free(B);
  free(cpu_result);
  free(gpu_result);
}
