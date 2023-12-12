#include "loss_functions.cuh"

#include "activation_functions.cuh"

#include <stdio.h>
#include <math.h>

/**
 * Cross-entropy loss function for classification problems
 */
float cross_entropy_loss(float* predicted, float *actual, size_t num_classes) {
  // Apply softmax to predictions
  softMax(predicted, num_classes);

  float loss = 0.0f;

  for (int i = 0; i < num_classes; i++) {
    loss += actual[i] - logf(predicted[i]);
  }

  return -loss;
}

/**
 * Squared error loss function for regression problems
 */
float squared_error_loss(float* predicted, float* actual, size_t size) {
  float loss = 0.0f;

  for (int i = 0; i < size; i++) {
    // The 1/2 term is to cancel out the derivative constant
    loss += (1.0f/ 2) * powf(predicted - actual, 2);
  }
  
  return loss;
}

/**
 * Derivative of the cross-entropy loss function
 * NOTE: This is only correct when softmax is used as the activation function
 */
float cross_entropy_loss_derivative(float predicted, float actual) {
  return predicted - actual;
}

/**
 * Derivative of the squared error loss function
 */
float squared_error_loss_derivative(float predicted, float actual) {
  return predicted - actual;
}

/** 
 * Convert loss_function_t enum to string representation4
 */
void loss_func_to_string(loss_func_t loss_func) {
  if (loss_func == CROSS_ENTROPY_LOSS) {
    printf("Cross-Entropy Loss\n");
  } else if (loss_func == SQUARED_ERROR_LOSS) {
    printf("Mean Squared Error Loss\n");
  }
}