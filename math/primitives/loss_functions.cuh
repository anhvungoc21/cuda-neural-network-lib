#pragma once

#include <stdlib.h>

// Enum for loss functions supplied by this library
typedef enum loss_func { CROSS_ENTROPY_LOSS, MSE_LOSS } loss_func_t;

// Cross-entropy loss for categorical problems
float cross_entropy_loss();

// Mean squared error loss for regression problems
float mean_squared_error_loss();
