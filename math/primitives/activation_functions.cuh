#pragma once

// Enum for activation functions supplied by this library
typedef enum activation_func { SIGMOID, RELU, SOFTMAX } activation_func_t;

// Sigmoid activation function
float sigmoid(float x);

// Rectified Linear Unit activation function
float reLU(float x);

// Softmax activation function
// This is currently only used internally when calculating cross-entropy loss
void softMax(float *x, size_t size);

// Derivative of Sigmoid activation function
float sigmoid_derivative(float output);

// Derivative of ReLU activation function
float reLU_derivative(float output);

// Convert activation_func_t enum to string representation
void activation_func_to_string(activation_func_t acti_func);