#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "math/math.cuh"
#include "math/primitives/activation_functions.cuh"
#include "math/primitives/loss_functions.cuh"

#include "network/network.cuh"

#include "utils/utils.cuh"

int main() {
  // Initialize network
  network_t *network = (network_t *) malloc(sizeof(network_t));
  size_t num_layers = 4;
  size_t num_inputs = 100;
  size_t num_outputs = 2;
  size_t num_epochs = 10;
  float lrate = 0.0004f;
  loss_func_t loss_func = CROSS_ENTROPY_LOSS;
  initialize_network(network, num_layers, num_inputs, num_outputs, num_epochs, lrate, loss_func);

  // Create layers
  create_append_layer(network, num_inputs, RELU); // 1 x 10
  create_append_layer(network, 120, RELU); // 10 x 8
  create_append_layer(network, 40, RELU); // 8 x 4
  create_append_layer(network, num_outputs, SIGMOID); // 4 x 2
  validate_network_arch(network);

  // Feed network data
  float *data = (float *) malloc(sizeof(float) * num_inputs);
  fill_random_floats(data, num_inputs);
  feed_input_data(network, data);

  // Forward propagate
  forward_propagate(network);

  // Print network
  print_network(network, true);
}
