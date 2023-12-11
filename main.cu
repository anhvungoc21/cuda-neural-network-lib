#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "math/math.cuh"

extern "C" {
#include "utils/utils.h"
#include "network/network.h"
#include "math/primitives/loss_functions.h"
#include "math/primitives/activation_functions.h"
}


int main() {
  network_t *network = (network_t *) malloc(sizeof(network_t));
  size_t num_layers = 3;
  size_t num_inputs = 4;
  size_t num_outputs = 2;
  size_t num_epochs = 10;
  float lrate = 0.0004f;
  loss_func_t loss_func = CROSS_ENTROPY_LOSS;
  initialize_network(network, num_layers, num_inputs, num_outputs, num_epochs, lrate, loss_func);

  create_append_layer(network, 1, RELU);
  create_append_layer(network, 3, RELU);
  create_append_layer(network, num_outputs, SIGMOID);

  print_arr(network->layers[0]->weights, network->layers[0]->prev_layer_dim * network->layers[0]->num_neurons);
  print_arr(network->layers[0]->biases, network->layers[0]->num_neurons);
  print_arr(network->layers[1]->weights, network->layers[1]->prev_layer_dim * network->layers[1]->num_neurons);
  print_arr(network->layers[1]->biases, network->layers[1]->num_neurons);
  print_arr(network->layers[2]->weights, network->layers[2]->prev_layer_dim * network->layers[2]->num_neurons);
  print_arr(network->layers[2]->biases, network->layers[2]->num_neurons);

  validate_network_arch(network);
}
