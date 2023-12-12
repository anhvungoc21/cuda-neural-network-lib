#include "loss_functions.cuh"

#include <stdio.h>

float cross_entropy_loss() { return 0.0f; }

float mean_squared_error_loss() { return 0.0f; }

/** 
 * Convert loss_function_t enum to string representation4
 */
void loss_func_to_string(loss_func_t loss_func) {
  if (loss_func == CROSS_ENTROPY_LOSS) {
    printf("Cross-Entropy Loss\n");
  } else if (loss_func == MSE_LOSS) {
    printf("Mean Squared Error Loss\n");
  }
}