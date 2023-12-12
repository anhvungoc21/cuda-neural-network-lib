#pragma once

// Enum for activation functions supplied by this library
typedef enum activation_func { SIGMOID, RELU } activation_func_t;

// Sigmoid activation function
float sigmoid(float x);

// Rectified Linear Unit activation function
float reLU(float x);

// Convert activation_func_t enum to string representation
void activation_func_to_string(activation_func_t acti_func);