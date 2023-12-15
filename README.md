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

Functions 1 through 6 have been implemented. Functions 7 through 10 are works in progress. Though, the current progress and future plans will be discussed in a later section.

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
9. `save_network()` (Not implemented):
Saves a network architecture and weights to a custom binary file.
10. `load_network()` (Not implemented):
Loads a network architecture and weights from a custom binary file to create a network.

## Demo Program
This section walks through the usage of a program to demonstrate the general instructions and example scenarios in using the library.

1. Normal Run

To start off, run `make main` to compile the demo program and run `./main`. This should successfully initializes the network, create the appropriate layers, validate the network, feed input data and perform one instance of forward propagation. The result seen should be a summary of the total time it took for the CPU and the GPU to perform forward propagation comparatively. The CPU should take a time in the order of seconds, while the GPU fractions of a second. After that, there should be a description of each layer architecture matching the specifications (`num_inputs` for the first layer, the intermediate layer dimensions, and `num_outputs` for the last layer) specified from lines 15 through 28. To verify that the matrix multiplications produced outputs of the correct dimensions, change the boolean on line 54 from `false` to `true`, which is a verbosity flag for printing. Compile and run the program again, which should now produce matrices of weights of the correct dimension. Since the dimensions in this case are incredibly big, this will prove more useful in the following examples.

2. Invalid Network Run

To test the correct initialization and validation of the network, intentionally do one of the following three things to make the architecture incorrect:
- On line 25, change `num_inputs` to an arbitrary number. This will cause a mismatch in the network specification and the actual dimensions of the input layer.
- On line 28, change `num_outputs` to an arbitrary number. This will cause the same mismatch, but for the output layer.
- From line 25 through 28, comment out one or more of the four lines. This will cause a mismatch in the number of layers in the network.
Each case should cause the program to spit out a helpful error message and exit. These originate from a call to `validate_network_arch()`.

3. Experimenting with CPU and GPU runtime

There are three salient cases of CPU vs. GPU runtime behavior when dealing with different input sizes:

- Input is too small: Both CPU and GPU takes a trivial amount of time to complete (~0s).
- Input is sufficiently small: CPU takes a short time to complete, while GPU incurs the cost of memory copying and end up performing worse.
- Input is sufficiently large: CPU takes a long time to complete (seconds), while GPU takes around the same order of time as they did for the previous case (fractions of seconds).

These cases correspond with the three commented-out values on lines 17, 26 and 27. This means that to test out case one, use the first set of values for each line starting from the left, and so on. Note that the value in the initial code is the third case. As a side note, like previously stated, using smaller values should allow you to verify the network architecture more acurately.

## Limitations and Future Directions
