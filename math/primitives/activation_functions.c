#include "activation_functions.h"

#include <math.h>

/**
 * Sigmoid activation function
 */
float sigmoid(float x) {
  // Handle too big/small input
  if (x >= 45.0f) return 1.0f;
  if (x <= -45.0f) return 0.0f;

  return 1.0f / (1.0f + exp(-x));
}

/**
 * Rectified Linear Unit activation function
 */
float reLU(float x) {
  return fmaxf(0.0f, x);
}