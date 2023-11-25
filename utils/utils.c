#include "utils.h"

/**
 * Fills an array with random whole-number floats from 1 to 10
 */
void fill_random(float *arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = (float)(rand() % 10 + 1);
  }
}

/**
 * Pretty prints a matrix
 */
void print_matrix(float *mat, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      printf("%.1f ", mat[r * cols + c]);
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
 * Checks 2 arrays for equality
 */
void check_equal_arr(float * arr1, float* arr2, int size) {
  for (int i = 0; i < size; i++) {
    if (arr1[i] != arr2[i]) {
      printf("Unequal: %f vs. %f", arr1[i], arr2[i]);
      return;
    }
  }

  printf("Equal!");
}