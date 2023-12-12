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
  size_t num_inputs = 8; // 8 // 1024 // 100000;
  size_t num_outputs = 2;
  size_t num_epochs = 10;
  float lrate = 0.0004f;
  loss_func_t loss_func = CROSS_ENTROPY_LOSS;
  initialize_network(network, num_layers, num_inputs, num_outputs, num_epochs, lrate, loss_func);

  // Create layers
  create_append_layer(network, num_inputs, RELU);
  create_append_layer(network, 16, RELU); // 16 // 512 // 2048
  create_append_layer(network, 4, RELU); // 4 // 128 // 10240
  create_append_layer(network, num_outputs, SIGMOID);

  // Validate architecture
  if (!validate_network_arch(network)) {
    exit(1);
  }

  // Print network
  // print_network(network, false);

  // Feed network data
  float *data = (float *) malloc(sizeof(float) * num_inputs);
  fill_random_floats(data, num_inputs);
  feed_input_data(network, data);

  // Forward propagate
  // CPU
  clock_t start_cpu = clock();
  forward_propagate(network, true);
  clock_t end_cpu = clock();
  printf("CPU: %.3f\n", (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC);

  // GPU
  clock_t start_gpu = clock();
  forward_propagate(network, false);
  clock_t end_gpu = clock();
  printf("GPU: %.3f\n", (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC);

  // Print network
  print_network(network, true);
}