#include "network.cuh"

#include "../math/math.cuh"
#include "../math/primitives/activation_functions.cuh"
#include "../math/primitives/loss_functions.cuh"

#include "../utils/utils.cuh"

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

// Hoist
void __initialize_weights_biases(layer_t *layer);

/**
 * Initializes a network with a set of predetermined specifications/settings.
 * These parameters are the ones that the network SHOULD have, 
 * not that it necessarily does at any given point in time.
 * 
 * \param network Pointer to a network_t struct
 * \param num_layers Number of layers
 * \param num_inputs Size of input layer
 * \param num_outputs Size of output layer
 * \param num_epochs Number of epochs to train the network
 * \param lrate Learning rate of network during training
 * \param loss_func Loss function that network should use
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
 * Checks whether the current network architecture aligns
 * with its user-defined specifications
 * 
 * \param network Pointer to a network_t struct
 *
 * \returns Whether network's architecture is valid (bool)
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
 * Creates a layer with a specified set of parameters
 * and append it to a designated network
 *
 * \param network Pointer to a network_t struct
 * \param num_neurons Number of neurons in the layer
 * \param act_func Activation function that the layer should use
 */
void create_append_layer(network_t *network, size_t num_neurons,
                         activation_func_t act_func) {
  // Create layer struct and initialize fields
  layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
  layer->num_neurons = num_neurons;
  layer->activation_func = act_func;

  // Store info about previous layer's output dimension
  // NOTE: For now, each neuron has 1 output, hence layer dimension = 1 x num_neurons
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

/**
 * Initializes the weights and biases of a layer
 *
 * Note: The current method is rudimentary and can cause symmetric weights & biases.
 * Other methods have been explored, but is difficult to implement:
 * - ReLU: He initialization
 * - Sigmoid: Xavier/Glorot initialization
 * https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
 * 
 * @param layer Pointer to a layer_t struct
 */
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
 * Feeds input data into the input layer of a network
 * 
 * Note: This currently allows 1 data point for each neuron.
 * Allowing each neuron to be a vector is the more common implementation.
 * 
 * \param network Pointer to a network_t struct
 * \param data An array of data
 */
void feed_input_data(network_t *network, float *data) {
  memcpy(network->layers[0]->outputs, data,
         sizeof(float) * network->num_inputs);
}

/**
 * Performs forward propagation through the network
 * 
 * \param network Pointer to a network_t struct
 * \param force_use_cpu Tell the program to must use CPU instead
 * of trying to determine between CPU and GPU based on fixed thresholds
 */
void forward_propagate(network_t *network, bool force_use_cpu) {
  // Propagate through each layer
  for (int i = 1; i < network->num_layers; i++) {
    layer_t *prev_layer = network->layers[i - 1];
    layer_t *cur_layer = network->layers[i];

    // Perform matrix multiplication of neuron data
    // - Matrix A: Dimensions of previous layer
    // - Matrix B: Number of previous neurons x Number of current neurons
    matrix_multiply(prev_layer->outputs, cur_layer->weights, cur_layer->outputs,
                    1, prev_layer->num_neurons, cur_layer->prev_layer_dim,
                    cur_layer->num_neurons, force_use_cpu);

    // Add biases
    for (int i = 0; i < cur_layer->num_neurons; i++) {
      cur_layer->outputs[i] += cur_layer->biases[i];
    }

    // Apply activation function
    activate_arr(cur_layer->outputs, cur_layer->num_neurons,
      cur_layer->activation_func);
  }
}

/**
 * Performs back propagation through the network 
 * and updates weights and biases for learning
 * 
 * TODO: UNFINISHED
 * 
 * \param network Pointer to a network_t struct
 * \param ground_truth Expected data to compare with predicted data for loss
 * \param force_use_cpu Tell the program to must use CPU instead
 *  of trying to determine between CPU and GPU based on fixed thresholds 
 */
void back_propagate(network_t *network, float *ground_truth, bool force_use_cpu) {
  for (int i = network->num_layers - 1; i > 0; i--) {
    layer_t *cur_layer = network->layers[i];
    // layer_t *next_layer = network->layers[i + 1];

    // At output layer: Calculate initial gradient of loss
    if (i == network->num_layers - 1) {
      // Gradient = Derivative of loss function * Derivative of activation function
      for (int i = 0; i < cur_layer->num_neurons; i++) {
        float d_loss_func = derivative_loss_func(ground_truth[i], cur_layer->outputs[i], network->loss_func);
        float d_acti_func = derivative_acti_func(cur_layer->outputs[i], cur_layer->activation_func);

        // Store gradient in outputs since we don't use this again
        cur_layer->outputs[i] = d_loss_func * d_acti_func;
      }
    }

    // TODO: Propagate gradient

  }  
}

/**
 * Saves the architecture, weights, and biases 
 * of a neural network to a custom binary file
 * 
 * TODO: UNFINISHED
 * 
 * \param network Pointer to a network_t struct
 * \param fname File name to save to
 */
void save_network(network_t *network, const char *fname) {
  FILE *file = fopen(fname, "wb");
  if (file == NULL) {
    perror("Error opening file for writing");
    return;
  }

  // Write network architecture data
  fwrite(&network->num_layers, sizeof(size_t), 1, file);
  fwrite(&network->num_inputs, sizeof(size_t), 1, file);
  fwrite(&network->num_outputs, sizeof(size_t), 1, file);
  fwrite(&network->num_epochs, sizeof(size_t), 1, file);
  fwrite(&network->learning_rate, sizeof(float), 1, file);
  fwrite(&network->loss_func, sizeof(int), 1, file);
  
  // TODO: Implement
}

/**
 * Prints network information layer by layer
 * 
 * \param network Pointer to a network struct
 * \param verbose Whether floats (weights, biases, outputs) should be printed
 */
void print_network(network_t *network, bool verbose) {
  for (int i = 0; i < network->num_layers; i++) {
    printf("\n========== LAYER %d ==========\n", i + 1);
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

      printf("Neuron Outputs:\n");
      print_arr(cur_layer->outputs, cur_layer->num_neurons);
    }
  }
}