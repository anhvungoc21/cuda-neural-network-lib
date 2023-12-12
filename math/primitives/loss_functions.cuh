#pragma once

#include <stdlib.h>

// Enum for loss functions supplied by this library
typedef enum loss_func { CROSS_ENTROPY_LOSS, SQUARED_ERROR_LOSS } loss_func_t;

// Cross-entropy loss function for classification problems
float cross_entropy_loss(float* predicted, float *actual, size_t num_classes);

// Squared error loss function for regression problems
float squared_error_loss(float* predicted, float* actual, size_t size);

// Derivative of the cross-entropy loss function
float cross_entropy_loss_derivative(float predicted, float actual);

// Derivative of the squared error loss function
float squared_error_loss_derivative(float predicted, float actual);

// Convert loss_function_t enum to string representation
void loss_func_to_string(loss_func_t loss_func);