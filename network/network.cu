#include "network.cuh"

#include "../math/math.cuh"
#include "../math/primitives/activation_functions.cuh"
#include "../math/primitives/loss_functions.cuh"

#include "../utils/utils.cuh"

#include <math.h>
#include <stdio.h>
#include <time.h>

// Hoist
void __initialize_weights_biases(layer_t *layer);

/**
 * Initializes a network with specified fields
 */
void initialize_network(network_t *network, size_t num_layers,
                        size_t num_inputs, size_t num_outputs,
                        size_t num_epochs, float lrate, loss_func_t loss_func) {
  // Initialize network struct fields
  network->num_layers = num_layers;
  network->num_inputs = num_inputs;
  network->num_outputs = num_outputs;
  network->num_epochs = num_epochs;
  network->learning_rate = lrate;
  network->loss_func = loss_func;

  // Allocate memory for layers
  network->layers = (layer_t **)malloc(sizeof(layer_t *) * num_layers);
  network->num_cur_layers = 0;
}

/**
 * Checks whether current network architecture aligns with its
 * user-defined specifications
 */
bool validate_network_arch(network_t *network) {
  // Check correctness of number of layers
  if (network->num_layers != network->num_cur_layers) {
    printf("Current number of layers (%d) does not match with specification "
           "(num_layers: %d)\n",
           network->num_cur_layers, network->num_layers);
    return false;
  }

  // Check correctness of input dimension
  if (network->layers[0]->num_neurons != network->num_inputs) {
    printf("Dimension of input layer (%d) does not match with specification "
           "(num_inputs: %d)\n",
           network->layers[0]->num_neurons, network->num_inputs);
    return false;
  }

  // Check correctness of output dimension
  if (network->layers[network->num_layers - 1]->num_neurons !=
      network->num_outputs) {
    printf("Dimension of output layer (%d) does not match with specification "
           "(num_outputs: %d)\n",
           network->layers[network->num_layers - 1]->num_neurons,
           network->num_outputs);
    return false;
  }

  return true;
}

/**
 * Creates a layer and append it to a designated network
 */
void create_append_layer(network_t *network, size_t num_neurons,
                         activation_func_t act_func) {
  // Create layer struct and initialize fields
  layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
  layer->num_neurons = num_neurons;
  layer->activation_func = act_func;

  // Store info about previous layer's output dimension
  if (network->num_cur_layers == 0) {
    layer->prev_layer_dim = 1;
  } else {
    layer->prev_layer_dim =
        (network->layers[network->num_cur_layers - 1])->num_neurons;
  };

  // Initialize outputs to all 0s
  layer->outputs = (float *)malloc(sizeof(float) * num_neurons);

  // Allocate space and Initialize weights and biases
  layer->weights = (float *)malloc(
      sizeof(float) * (layer->prev_layer_dim * layer->num_neurons));
  layer->biases = (float *)malloc(sizeof(float) * layer->num_neurons);
  if (network->num_cur_layers != 0) {
    __initialize_weights_biases(layer);
  }

  // Append layer to network
  network->layers[network->num_cur_layers] = layer;
  network->num_cur_layers++;
}

void __initialize_weights_biases(layer_t *layer) {
  srand((unsigned int)time(NULL));

  // Initialize weights to random values from -1.0 to 1.0
  for (int i = 0; i < layer->prev_layer_dim * layer->num_neurons; i++) {
    int randInt = rand();
    double randFloat = ((float)randInt / RAND_MAX) * 2.0f - 1.0f;
    layer->weights[i] = randFloat;
  }

  // Initialize biases to 0s
  for (int i = 0; i < layer->num_neurons; i++) {
    layer->biases[i] = 0.0f;
  }
}

/**
 * Store input data into input layer of network
 */
void feed_input_data(network_t *network, float *data) {
  memcpy(network->layers[0]->outputs, data,
         sizeof(float) * network->num_inputs);
}

/**
 * Perform forward propagation through the network
 */
void forward_propagate(network_t *network) {
  for (int i = 1; i < network->num_layers; i++) {
    layer_t *prev_layer = network->layers[i - 1];
    layer_t *cur_layer = network->layers[i];

    // Perform matrix multiplication of neuron data
    // Layer always has dim 1 x num_neurons, hence 1
    matrix_multiply(prev_layer->outputs, cur_layer->weights, cur_layer->outputs,
                    1, prev_layer->num_neurons, cur_layer->prev_layer_dim,
                    cur_layer->num_neurons, false);

    // Apply activation function
    activate_arr(cur_layer->outputs, cur_layer->num_neurons,
                 cur_layer->activation_func);

    // Add biases
    for (int i = 0; i < cur_layer->num_neurons; i++) {
      cur_layer->outputs[i] += cur_layer->biases[i];
    }
  }
}

/**
 * Print network info layer by layer
 * If verbose, print weights and biases
 */
void print_network(network_t *network, bool verbose) {
  for (int i = 0; i < network->num_layers; i++) {
    printf("========== LAYER %d ==========\n", i + 1);
    layer_t *cur_layer = network->layers[i];

    printf("Number of neurons: %d\n", cur_layer->num_neurons);
    printf("Activation function: ");
    activation_func_to_string(cur_layer->activation_func);

    if (verbose) {
      printf("Weights:\n");
      print_matrix(cur_layer->weights, cur_layer->prev_layer_dim,
                   cur_layer->num_neurons);

      printf("Biases:\n");
      print_arr(cur_layer->biases, cur_layer->num_neurons);
    }

    printf("Neuron Outputs:\n");
    print_arr(cur_layer->outputs, cur_layer->num_neurons);
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