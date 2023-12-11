#include "network.h"

#include "../math/primitives/loss_functions.h"
#include "../math/primitives/activation_functions.h"

#include <math.h>
#include <time.h>
#include <stdio.h>

// Hoist
void __initialize_weights_biases(layer_t *layer);

/**
 * Initializes a network with specified fields 
 */
void initialize_network(network_t *network, size_t num_layers, size_t num_inputs, size_t num_outputs, size_t num_epochs, float lrate, loss_func_t loss_func) {
  // Initialize network struct fields
  network->num_layers = num_layers;
  network->num_inputs = num_inputs;
  network->num_outputs = num_outputs;
  network->num_epochs = num_epochs;
  network-> learning_rate = lrate;
  network->loss_func = loss_func;

  // Allocate memory for layers
  network->layers = malloc(sizeof(layer_t*) * num_layers);
  network->num_cur_layers = 0;
}

/**
 * Checks whether current network architecture aligns with its
 * user-defined specifications
 */
bool validate_network_arch(network_t *network) {
  // Check correctness of number of layers
  if (network->num_layers != network->num_cur_layers) {
    printf("Current number of layers (%d) does not match with specification (num_layers: %d)\n", network->num_cur_layers, network->num_layers);
    return false;
  }

  // Check correctness of input dimension
  if (network->layers[0]->num_neurons != network->num_inputs) {
    printf("Dimension of input layer (%d) does not match with specification (num_inputs: %d)\n", network->layers[0]->num_neurons, network->num_inputs);
    return false;
  }

  // Check correctness of output dimension
  if (network->layers[network->num_layers - 1]->num_neurons != network->num_outputs) {
    printf("Dimension of output layer (%d) does not match with specification (num_outputs: %d)\n", network->layers[network->num_layers - 1]->num_neurons, network->num_outputs);
    return false;
  }

  return true;
}

/**
 * Creates a layer and append it to a designated network
 */
void create_append_layer(network_t *network, size_t num_neurons, activation_func_t act_func) {
  // Create layer struct and initialize fields
  layer_t *layer = malloc(sizeof(layer_t));
  layer->num_neurons = num_neurons;
  layer->activation_func = act_func;

  // Store info about previous layer's output dimension
  if (network->num_cur_layers == 0) {
    layer->prev_layer_dim = 1;
  } else {
    layer->prev_layer_dim = (network->layers[network->num_cur_layers - 1])->num_neurons;
  };

  // Initialize outputs to all 0s
  layer->outputs = malloc(sizeof(float) * num_neurons);
  
  // Allocate space and Initialize weights and biases
  layer->weights = malloc(sizeof(float) * (layer->prev_layer_dim * layer->num_neurons));
  layer->biases = malloc(sizeof(float) * layer->num_neurons);
  __initialize_weights_biases(layer);

  // Append layer to network
  network->layers[network->num_cur_layers] = layer;
  network->num_cur_layers++;
}

void __initialize_weights_biases(layer_t *layer) {
  srand((unsigned int)time(NULL));

  // Initialize weights to random values from -1.0 to 1.0
  for (int i = 0; i < layer->prev_layer_dim * layer->num_neurons; i++) {
    int randInt = rand();
    double randFloat = ((float) randInt / RAND_MAX) * 2.0f - 1.0f;
    layer->weights[i] = randFloat;
  }

  // Initialize biases to 0s
  for (int i = 0; i < layer->num_neurons; i++) {
    layer->biases[i] = 0.0f;
  }
}

// Notes about Dimensions:
// Weights of a layer:
// - Rows: prev_layer_dim
// - Cols: num_neurons

// Possible improvements:
// - Initialize weights using valid strategy:
//   + ReLU: He initialization
//   + Sigmoid: Xavier/Glorot initialization
// - Initialize weights with GPU