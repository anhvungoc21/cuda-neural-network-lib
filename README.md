# GPU-Accelerated Neural Network Library
> A library for creating GPU-accelerated neural networks written in CUDA C.

## Repository Structure
Below is a description of the relevant files and directories and the role they play in the project:

`main.cu`: This is the entrypoint to a program to test out features of the project. This can be compiled with `make main`, which will compile all project files and produce an executable called `main`.

`network/`: This subdirectory contains the implementation of the neural network primitives and functions. The structs and functions that should be exposed to the user of this library are listed in `network.cuh`.

`math/`: This subdirectory contains the implementation of all math-related portions of the project:
- `math.cuh`: Contains functions that should be used by the network. This module abstracts `cpu_math.cuh` and `gpu_math.cuh`, which contain duplicated implementations of functions, but utilizing the CPU and GPU respectively.
- `primitives/`: This inner subdirectory contains implementations of math equations which are derived straight from calculus. Inside, there are implementations of activation functions, loss functions, and their derivatives.

`utils/`: This subdirectory contains miscellaneous utilities that can be used at multiple places in the project, such as pretty printing, random array generation, and error checking for CUDA calls.

## Use of System
The intended audience of this project are developers interested in neural networks, hence usage will involve writing code to produce neural networks. This section will provide a walkthrough of the core series of functions (both implemented and in progress) to create a neural network, while the next section will provide an instruction to compile and run a demo program.

Functions 1 through 6 have been implemented. Functions 7 and 8 works in progress. Though, the current progress and future plans will be discussed in a later section.

1. `initialize_network()`: Takes in a pointer to a malloc-ed network and a set of parameters, and initialize a network of type `network_t`. This struct can be seen in `network/network.cuh`.

2. `create_layer()`: Takes in a pointer to a network and a set of parameters to create a layer of type `layer_t` with those parameters and append it to the network's architecture. This struct can be seen in `network/network.cuh`.

3. `validate_network()`: Before anything is done with a network, a user should call this function on the network to verify that the added layers satisfy the set parameters in `initialize_network()`.

4. `feed_input_data()`: Simply feeds an array of data points into the first layer of the network. Each neuron receives one data point.

5. `forward_propagate()`: Propagates input data forward through each layer of the network, performing matrix multiplications on a layer's weights, adding its biases, and applying its activation function to get the layer's neurons' outputs. Here:
- `matrix_multiply()` is called, which is the crux of the usage of GPU thread parallelism and thread synchronization. More details on this later.
- The addition of biases and application of activation functions could (and should) also be implemenented on the GPU, but I only had time to implement them on the CPU. To keep the code modular, I plan to have each step be a function that outputs and reuses pointers to GPU memory, instead of merging all three steps into a single GPU kernel.
6. `print_network()`: Prints a network architecture and its layers' architectures. If a verbose flag is set to true, the weights, biases, and neuron outputs are also printed.
7. `back_propagate()` (Partially implemented): Propagates loss in outputs at the final layer back through the network, mainly performing matrix multiplications and derivatives of activation functions. At each layer, the error corresponding to each neuron is calculated, which is used to adjust its weights and bias term.
8. `train_network()` (Not implemented): Runs a combination of forward propagation and back-propagation for a set number of epochs to perform gradient descent and consequently train the network.

