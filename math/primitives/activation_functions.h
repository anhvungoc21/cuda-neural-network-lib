#pragma once

// Enum for activation functions supplied by this library
typedef enum activation_func {
  SIGMOID,
  RELU
} activation_func_t;

// Sigmoid activation function
float sigmoid(float x);

// Rectified Linear Unit activation function
float reLU(float x);