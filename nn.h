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
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>

// You can change this to float, if you want
typedef double nn_float_t;

// Activation functions
enum {
    NN_LINEAR,  // y = x
    NN_RELU,    // y = max(0, x)
    NN_SIGMOID  // y = 1 / (1 + exp(-x))
};

typedef struct {
    nn_float_t *neurons;  // neurons on this layer
    nn_float_t *deltas;   // derivative of the error wrt each neuron's input (NULL for input layer)
    nn_float_t *weights;  // (neuronCnt + 1) * nextLayer.neuronCnt (NULL for output layer)

    uint32_t neuronCnt;  // number of neurons on this layer
    uint32_t actId;      // identifier of activation function (ignored for input layer)
} nn_layer_t;

typedef struct {
    // For best performance, allocate everything in one memory block. Layout is:
    // - nn_float_t weights[weightCnt]
    // - nn_float_t neurons[neuronCnt]
    // - nn_float_t deltas[neuronCnt - layers[0].neuronCnt] (input layer has no deltas)
    nn_float_t *block;

    // To reduce indexing hell, layers[] contain what is needed to handle block[] data easily
    nn_layer_t *layers;

    uint32_t layerCnt, weightCnt, neuronCnt, pad;
} nn_network_t;

// Create a network (zero initialized)
nn_network_t nn_network_init(
    uint32_t layerCnt,     // number of layers (including input and output)
    uint32_t *neuronCnts,  // layerCnt elements
    uint32_t *actIds);     // layerCnt - 1 elements (input layer has no activation)

// Releases memory (and sets *nn = {0} for good measure)
void nn_network_destroy(nn_network_t *nn);

// Print an array of n elements (useful for debugging)
void nn_array_print(size_t n, const nn_float_t *array);

// Print network, or layer (useful for debugging). 'what' is a combination of characters:
// - 'a': print the activation function (not applicable for the input layer)
// - 'n': print the neurons
// - 'd': print the deltas (not applicable for the input layer)
// - 'w': print the weights (not applicable for the output layer)
void nn_layer_print(
    const nn_layer_t *layer,
    uint32_t nextLayerNeuronCnt,
    const char *what);

void nn_network_print(const nn_network_t *nn,
    const char *what);

// Run the network forward. If inputs == NULL, use those already stored in nn->layers[0].neurons[].
void nn_run(const nn_network_t *nn,
    const nn_float_t *inputs);  // nn->layers[0].neuronCnt elements

// Run the network forward, and compute the deltas[] based on a sample = (inputs, outputs)
void nn_backprop(const nn_network_t *nn,
    const nn_float_t *inputs,   // nn->layers[0].neuronCnt elements (or NULL)
    const nn_float_t *outputs,  // nn->layers[nn->layerCnt - 1].neuronCnt elements
    bool absolute);             // use absolute error |x-y| (otherwise squared 0.5*(x-y)^2)

// Same as backprop, but goes a step further to retreive the gradient
void nn_gradient(const nn_network_t *nn,
    const nn_float_t *inputs,   // nn->layers[0].neuronCnt elements (or NULL)
    const nn_float_t *outputs,  // nn->layers[nn->layers - 1].neuronCnt elements
    nn_float_t *gradient,       // nn->weightCnt elements
    bool absolute);             // use absolute error |x-y| (otherwise squared 0.5*(x-y)^2)

// Save network to file (binary format)
void nn_save(const nn_network_t *nn,
    FILE *out);

// Load network from file (binary format). This instantiates a new network (allocates memory).
nn_network_t nn_load(FILE *in);
