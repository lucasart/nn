/*
 * nn, a simple neuron network framework in C. Copyright 2021 lucasart.
 *
 * c-chess-cli is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * c-chess-cli is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program. If
 * not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include <stdbool.h>
#include <inttypes.h>

// Activation functions
typedef double(*nn_func_t)(double);

enum {
    NN_LINEAR,  // y = x
    NN_RELU,  // y = max(0, x)
    NN_SIGMOID  // y = 1 / (1 + exp(-x))
};

typedef struct {
    double *neurons;  // neurons on this layer
    double *deltas;  // derivative of the error wrt each neuron's input (NULL for input layer)
    double *weights;  // (neuronCnt + 1) * nextLayer.neuronCnt (NULL for output layer)

    uint32_t neuronCnt;  // number of neurons on this layer
    uint32_t actId;  // identifier of activation function (eg. NN_RELU)
} nn_layer_t;

void nn_array_print(size_t n, const double *array);
void nn_layer_print(const nn_layer_t *layer, uint32_t nextLayerNeuronCnt, const char *what);

typedef struct {
    // For best performance, allocate everything in one memory block. Layout is:
    // - double weights[weightCnt]
    // - double neurons[neuronCnt]
    // - double deltas[neuronCnt - layers[0].neuronCnt] (input layer has no deltas)
    double *block;

    // To reduce indexing hell, layers[] contain what is needed to handle block[] data easily
    nn_layer_t *layers;

    uint32_t layerCnt, weightCnt, neuronCnt;
} nn_network_t;

// Create a network (zero initialized):
// - layerCnt: how many layers, including input and output layer (must be >= 2).
// - neuronCnts[layerCnt]: how many neurons per layer.
// - actIds[layerCnt - 1]: activation function for each layer (starts with layer 1, because layer 0
//   is just the input layer which has no activation function).
nn_network_t nn_network_init(uint32_t layerCnt, uint32_t *neuronCnts, uint32_t *actIds);

// Releases memory (and sets *nn = {0} for good measure)
void nn_network_destroy(nn_network_t *nn);

// Display the network in human readable form on stdout. what is a string of characters, used to say
// what you want to print:
// - 'a': print the activation function (not applicable for the first layer)
// - 'n': print the neurons
// - 'd': print the deltas (computed by backprop)
// - 'w': print the weights
void nn_network_print(const nn_network_t *nn, const char *what);

// Run the network forward. inputs may be:
// - double[nn->layers[0].neuronCnt]: copies the inputs[] array into nn->layers[0].neurons[].
// - NULL: assumes inputs are already placed in nn->layers[0].neurons[].
void nn_run(const nn_network_t *nn, const double *inputs);

// Run the network forward, and compute the deltas[] based on a sample = (inputs, outputs)
// absolute: if true, use absolute error |x-y|, else use squared error 0.5*(x-y)^2
void nn_backprop(const nn_network_t *nn, const double *inputs, const double *outputs,
    bool absolute);

// Same as backprop, but goes a step further to retreive the gradient[nn->weightCnt]
void nn_gradient(const nn_network_t *nn, const double *inputs, const double *outputs, bool absolute,
    double *gradient);
