#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Fills an array with random whole-number floats from 1 to 10
void fill_random_ints(float *arr, int size);

// Fills an array with random floats from 1 to 10
void fill_random_floats(float *arr, int size);

// Prints a matrix
void print_matrix(float *mat, int rows, int cols);

// Prints an array
void print_arr(float *arr, int size);

// Checks 2 arrays for equality
void check_equal_arr(float *arr1, float *arr2, int size, float epsilon);

// Checks whether 2 floats are almost equal
int almostEquals(float a, float b, float epsilon);