#include "activation_functions.cuh"

#include <math.h>
#include <stdio.h>

/**
 * Sigmoid activation function
 */
float sigmoid(float x) {
  // Handle too big/small input
  if (x >= 45.0f)
    return 1.0f;
  if (x <= -45.0f)
    return 0.0f;

  return 1.0f / (1.0f + exp(-x));
}

/**
 * Rectified Linear Unit activation function
 */
float reLU(float x) { return fmaxf(0.0f, x); }

/**
 * Softmax activation function
 */
void softMax(float *arr, size_t size) {
  float sum_exp = 0.0f;
  
  // Calculate and sum exponents
  for (int i = 0; i < size; i++) {
    arr[i] = exp(arr[i]);
    sum_exp += arr[i];
  }

  // Divide exponents by sum
  for (int i = 0; i < size; i++) {
    arr[i] /= sum_exp;
  }
}

/** 
 * Derivative of Sigmoid activation function
 */
float sigmoid_derivative(float output) {
  return output * (1.0f - output);
}

/** Derivative of ReLU activation function
 *
 */
float reLU_derivative(float output) {
  return (output > 0.0f) ? 1.0f : 0.0f;
}

/**
 * Translate activation_func_t enum to string representation
 */
void activation_func_to_string(activation_func_t acti_func) {
  if (acti_func == RELU) {
    printf("ReLU\n");
  } else if (acti_func == SIGMOID) {
    printf("Sigmoid\n");
  }
} 