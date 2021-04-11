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

// Activation functions
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
    uint32_t actId;  // identifier of activation function (ignored for input layer)
} nn_layer_t;

typedef struct {
    // For best performance, allocate everything in one memory block. Layout is:
    // - double weights[weightCnt]
    // - double neurons[neuronCnt]
    // - double deltas[neuronCnt - layers[0].neuronCnt] (input layer has no deltas)
    double *block;

    // To reduce indexing hell, layers[] contain what is needed to handle block[] data easily
    nn_layer_t *layers;

    uint32_t layerCnt, weightCnt, neuronCnt, pad;
} nn_network_t;

// Create a network (zero initialized):
// - layerCnt: how many layers, including input and output layer (must be >= 2).
// - neuronCnts[l]: how many neurons on layer l.
// - actIds[l]: activation function for layer l + 1 (input layer has none).
nn_network_t nn_network_init(uint32_t layerCnt, uint32_t neuronCnts[layerCnt],
    uint32_t actIds[layerCnt - 1]);

// Releases memory (and sets *nn = {0} for good measure)
void nn_network_destroy(nn_network_t *nn);

// Handy function to print an array (comma separated)
void nn_array_print(size_t n, const double array[n]);

// Display the network, or a layer, in human readable form. what is a combination of characters:
// - 'a': print the activation function (not applicable for the input layer)
// - 'n': print the neurons
// - 'd': print the deltas (not applicable for the input layer)
// - 'w': print the weights (not applicable for the output layer)
void nn_layer_print(const nn_layer_t *layer, uint32_t nextLayerNeuronCnt, const char *what);
void nn_network_print(const nn_network_t *nn, const char *what);

// Run the network forward. If inputs == NULL, use those already stored in nn->layers[0].neurons[].
void nn_run(const nn_network_t *nn, const double inputs[nn->layers[0].neuronCnt]);

// Run the network forward, and compute the deltas[] based on a sample = (inputs, outputs)
// absolute: if true, use absolute error |x-y|, else use squared error 0.5*(x-y)^2
void nn_backprop(const nn_network_t *nn, const double inputs[nn->layers[0].neuronCnt],
    const double outputs[nn->layers[nn->layerCnt - 1].neuronCnt], bool absolute);

// Same as backprop, but goes a step further to retreive the gradient[nn->weightCnt]
void nn_gradient(const nn_network_t *nn, const double inputs[nn->layers[0].neuronCnt],
    const double outputs[nn->layers[nn->layerCnt - 1].neuronCnt], bool absolute,
    double gradient[nn->weightCnt]);

// Save network to file (binary format)
void nn_save(const nn_network_t *nn, FILE *out);

// Load network from file (binary format). This instantiates a new network (allocates memory).
nn_network_t nn_load(FILE *in);
