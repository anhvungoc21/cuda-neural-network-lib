#include "utils.cuh"

/**
 * Fills an array with random whole-number floats from 1 to 10
 */
void fill_random_ints(float *arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = (float)(rand() % 10 + 1);
  }
}

/**
 * Fills an array with random floats from 1 to 10
 */
void fill_random_floats(float *arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = 1.0f + ((float)rand() / RAND_MAX) * 9.0f;
  }
}

/**
 * Pretty prints a matrix
 */
void print_matrix(float *mat, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      printf("%.3f ", mat[r * cols + c]);
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * Prints an array
 */
void print_arr(float *arr, int size) {
  for (int i = 0; i < size; i++) {
    printf("%.1f ", arr[i]);
  }
  printf("\n\n");
}

/**
 * Checks whether 2 floats are almost equal
 */
int almostEquals(float a, float b, float epsilon) {
  return fabs(a - b) < epsilon;
}

/**
 * Checks 2 arrays for equality
 */
void check_equal_arr(float *arr1, float *arr2, int size, float epsilon) {
  for (int i = 0; i < size; i++) {
    if (!almostEquals(arr1[i], arr2[i], epsilon)) {
      printf("Unequal: %.6f vs. %.6f\n", arr1[i], arr2[i]);
      return;
    }
  }

  printf("Equal!\n");
}