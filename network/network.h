#pragma once

#include <stdlib.h>
#include <stdbool.h>

#include "../math/primitives/activation_functions.h"
#include "../math/primitives/loss_functions.h"

/**
 * Struct for a layer in a neural network
 * \field num_neurons Number of neurons in this layer
 * \field outputs Output data stored in neurons
 * \field weights Weights of edges leading into each neuron
 * \field biases Bias terms at each neuron
 * 
 * \field activation_func Activation function for this layer
*/
typedef struct layer {
  // Architecture
  size_t num_neurons;
  float *outputs;
  float *weights;
  float *biases;

  // Meta
  activation_func_t activation_func;

  // Previous layer
  size_t prev_layer_dim;
} layer_t;

/**
 * Struct for a neural network
 * \field num_layers Number of layers in the network
 * \field layer Array of layers in the network
 * \field num_inputs Number of data points in the input layer
 * \field num_outputs Number of nodes in the output layer
 *
 * \field num_epochs Number of epochs used to train the network
 * \field learning_rate Learning rate used to update weights
 * \field loss_function Loss function used to measure prediction error
 */
typedef struct network {
  // Architecture
  size_t num_layers;
  layer_t **layers;
  size_t num_inputs;
  size_t num_outputs;

  // Meta
  size_t num_epochs;
  float learning_rate;
  loss_func_t loss_func;

  // Counter
  size_t num_cur_layers;
} network_t;

// Initializes a network with specified fields 
void initialize_network(network_t *network, size_t num_layers, size_t num_inputs, size_t num_outputs, size_t num_epochs, float lrate, loss_func_t loss_func);

// Creates a layer and append it to a designated network
void create_append_layer(network_t *network, size_t num_neurons, activation_func_t act_func);

// Checks whether current network architecture aligns with its user-defined specifications
bool validate_network_arch(network_t *network);