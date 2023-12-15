#include "cpu_math.cuh"

#include "primitives/activation_functions.cuh"
#include "primitives/loss_functions.cuh"

#include <stdio.h>

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

/**
 * Applies a specified activation on an array on the CPU
 * 
 * \param arr Array to activate element-wise
 * \param size Size of array
 * \param acti_func Activation function to use
 */
void cpu__activate_arr(float *arr, size_t size, activation_func_t acti_func) {
  for (int i = 0; i < size; i++) {
    if (acti_func == RELU) {
      arr[i] = reLU(arr[i]);
    } else if (acti_func == SIGMOID) {
      arr[i] = sigmoid(arr[i]);
    } else {
      printf("Activation function not currently supported!");
      exit(EXIT_FAILURE);
    }
  }
}

/**
 * Calculates the derivative of an activation function on the CPU
 * 
 * @param output Output of activation function
 * @param acti_func Activation function used
 *
 * @returns Derivative of activation function with such output
 */
float cpu__derivative_acti_func(float output, activation_func acti_func) {
  if (acti_func == RELU) {
    return reLU_derivative(output);
  } else if (acti_func == SIGMOID) {
    return sigmoid_derivative(output);
  } else {
    printf("Activation function not currently supported, let alone its derivative! :D");
    exit(EXIT_FAILURE);
  }
}

/**
 * Calculates a loss function for an array on the CPU
 * 
 * \param predicted Predicted output value
 * \param predicted Actual output value (ground truth)
 * \param loss_func Loss function to use
 *
 * \returns Loss calculated
 */
float cpu__calculate_loss(float *predicted, float *actual, size_t size, loss_func_t loss_func) {
  if (loss_func == CROSS_ENTROPY_LOSS) {
    return cross_entropy_loss(predicted, actual, size);
  } else if (loss_func == SQUARED_ERROR_LOSS) {
    return squared_error_loss(predicted, actual, size);
  } else {
    printf("Loss function not currently supported!");
    exit(EXIT_FAILURE);
  }
}

/**
 * Calculates the derivative of a loss function on the CPU
 * 
 * \param predicted Predicted output value
 * \param predicted Actual output value (ground truth)
 * \param loss_func Loss function used
 *
 * \returns Derivative of the loss function with such predicted and actual values
 */
float cpu__derivative_loss_func(float predicted, float actual, loss_func_t loss_func) {
  if (loss_func == CROSS_ENTROPY_LOSS) {
    return cross_entropy_loss_derivative(predicted, actual);
  } else if (loss_func == SQUARED_ERROR_LOSS) {
    return squared_error_loss_derivative(predicted, actual);
  } else {
    printf("Loss function not currently supported, let alone its derivative! :D");
    exit(EXIT_FAILURE);
  }
}